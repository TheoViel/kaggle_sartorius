import torch
import numpy as np

from mmdet.core import bbox2roi, bbox2result, merge_aug_masks
from mmdet.models.builder import HEADS, build_head, DETECTORS
from mmdet.models.roi_heads.cascade_roi_head import CascadeRoIHead
from mmdet.models.detectors.two_stage import TwoStageDetector


@DETECTORS.register_module()
class CascadeMaskScoringRCNN(TwoStageDetector):
    """Mask Scoring RCNN.
    https://arxiv.org/abs/1903.00241
    """

    def __init__(
        self,
        backbone,
        rpn_head,
        roi_head,
        train_cfg,
        test_cfg,
        neck=None,
        pretrained=None,
        init_cfg=None,
    ):
        super(CascadeMaskScoringRCNN, self).__init__(
            backbone=backbone,
            neck=neck,
            rpn_head=rpn_head,
            roi_head=roi_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            pretrained=pretrained,
            init_cfg=init_cfg,
        )


@HEADS.register_module()
class CascadeMaskScoringRoIHead(CascadeRoIHead):
    """
    Cascade + Mask Scoring RoIHead.
    Adapted from :
    https://github.com/open-mmlab/mmdetection/blob/master/mmdet/models/roi_heads/mask_scoring_roi_head.py
    """

    def __init__(
        self,
        mask_iou_head,
        num_stages,
        stage_loss_weights,
        bbox_roi_extractor=None,
        bbox_head=None,
        mask_roi_extractor=None,
        mask_head=None,
        shared_head=None,
        train_cfg=None,
        test_cfg=None,
        pretrained=None,
        init_cfg=None,
    ):
        assert mask_iou_head is not None

        super(CascadeMaskScoringRoIHead, self).__init__(
            num_stages,
            stage_loss_weights,
            bbox_roi_extractor=bbox_roi_extractor,
            bbox_head=bbox_head,
            mask_roi_extractor=mask_roi_extractor,
            mask_head=mask_head,
            shared_head=shared_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            pretrained=pretrained,
            init_cfg=init_cfg,
        )
        self.mask_iou_head = build_head(mask_iou_head)

    def _mask_forward_train(
        self, stage, x, sampling_results, gt_masks, rcnn_train_cfg, bbox_feats=None
    ):
        """
        Run forward function and calculate loss for Mask head in training.
        """
        # Cascade forward
        pos_rois = bbox2roi([res.pos_bboxes for res in sampling_results])

        mask_roi_extractor = self.mask_roi_extractor[stage]
        mask_head = self.mask_head[stage]

        mask_feats = mask_roi_extractor(x[: mask_roi_extractor.num_inputs], pos_rois)
        mask_pred = mask_head(mask_feats)

        # Cascade loss
        mask_targets = mask_head.get_targets(sampling_results, gt_masks, rcnn_train_cfg)
        pos_labels = torch.cat([res.pos_gt_labels for res in sampling_results])
        loss_mask = mask_head.loss(mask_pred, mask_targets, pos_labels)

        mask_results = dict(
            mask_pred=mask_pred,
            mask_feats=mask_feats,
            loss_mask=loss_mask,
            mask_targets=mask_targets
        )

        if mask_results["loss_mask"] is None:
            return mask_results

        # Mask IoU predictions for positive masks
        pos_labels = torch.cat([res.pos_gt_labels for res in sampling_results])

        pos_mask_pred = mask_pred[
            range(mask_pred.size(0)), pos_labels
        ]
        mask_iou_pred = self.mask_iou_head(mask_feats, pos_mask_pred)
        pos_mask_iou_pred = mask_iou_pred[range(mask_iou_pred.size(0)), pos_labels]

        # Mask IoU loss
        mask_iou_targets = self.mask_iou_head.get_targets(
            sampling_results,
            gt_masks,
            pos_mask_pred,
            mask_targets,
            rcnn_train_cfg,
        )

        loss_mask_iou = self.mask_iou_head.loss(pos_mask_iou_pred, mask_iou_targets)
        mask_results["loss_mask"].update(loss_mask_iou)

        return mask_results

    def simple_test(self, x, proposal_list, img_metas, rescale=False):
        """
        Test without augmentation.
        Adapted from :
        https://github.com/open-mmlab/mmdetection/blob/a7a16afbf2a4bdb4d023094da73d325cb864838b/mmdet/models/roi_heads/cascade_roi_head.py#L288
        """
        # Boxes
        assert self.with_bbox, "Bbox head must be implemented."
        num_imgs = len(proposal_list)
        img_shapes = tuple(meta["img_shape"] for meta in img_metas)
        ori_shapes = tuple(meta["ori_shape"] for meta in img_metas)
        scale_factors = tuple(meta["scale_factor"] for meta in img_metas)

        # "ms" in variable names means multi-stage
        ms_scores = []
        rcnn_test_cfg = self.test_cfg

        rois = bbox2roi(proposal_list)

        if rois.shape[0] == 0:
            # There is no proposal in the whole batch
            bbox_results = [
                [
                    np.zeros((0, 5), dtype=np.float32)
                    for _ in range(self.bbox_head[-1].num_classes)
                ]
            ] * num_imgs

            if self.with_mask:
                mask_classes = self.mask_head[-1].num_classes
                segm_results = [
                    [[] for _ in range(mask_classes)] for _ in range(num_imgs)
                ]
                results = list(zip(bbox_results, segm_results))
            else:
                results = bbox_results

            return results

        for i in range(self.num_stages):
            bbox_results = self._bbox_forward(i, x, rois)

            # split batch bbox prediction back to each image
            cls_score = bbox_results["cls_score"]
            bbox_pred = bbox_results["bbox_pred"]
            num_proposals_per_img = tuple(len(proposals) for proposals in proposal_list)
            rois = rois.split(num_proposals_per_img, 0)
            cls_score = cls_score.split(num_proposals_per_img, 0)
            if isinstance(bbox_pred, torch.Tensor):
                bbox_pred = bbox_pred.split(num_proposals_per_img, 0)
            else:
                bbox_pred = self.bbox_head[i].bbox_pred_split(
                    bbox_pred, num_proposals_per_img
                )
            ms_scores.append(cls_score)

            if i < self.num_stages - 1:
                if self.bbox_head[i].custom_activation:
                    cls_score = [
                        self.bbox_head[i].loss_cls.get_activation(s) for s in cls_score
                    ]
                refine_rois_list = []
                for j in range(num_imgs):
                    if rois[j].shape[0] > 0:
                        bbox_label = cls_score[j][:, :-1].argmax(dim=1)
                        refined_rois = self.bbox_head[i].regress_by_class(
                            rois[j], bbox_label, bbox_pred[j], img_metas[j]
                        )
                        refine_rois_list.append(refined_rois)
                rois = torch.cat(refine_rois_list)

        # average scores of each image by stages
        cls_score = [
            sum([score[i] for score in ms_scores]) / float(len(ms_scores))
            for i in range(num_imgs)
        ]

        # apply bbox post-processing to each image individually
        det_bboxes = []
        det_labels = []
        for i in range(num_imgs):
            det_bbox, det_label = self.bbox_head[-1].get_bboxes(
                rois[i],
                cls_score[i],
                bbox_pred[i],
                img_shapes[i],
                scale_factors[i],
                rescale=rescale,
                cfg=rcnn_test_cfg,
            )
            det_bboxes.append(det_bbox)
            det_labels.append(det_label)

        bbox_results = [
            bbox2result(det_bboxes[i], det_labels[i], self.bbox_head[-1].num_classes)
            for i in range(num_imgs)
        ]

        # Masks
        assert self.with_mask

        # No boxes
        if all(det_bbox.shape[0] == 0 for det_bbox in det_bboxes):
            mask_classes = self.mask_head[-1].num_classes
            segm_results = [
                [[] for _ in range(mask_classes)] for _ in range(num_imgs)
            ]
            return list(zip(bbox_results, segm_results))

        # Rescale boxes
        if rescale and not isinstance(scale_factors[0], float):
            scale_factors = [
                torch.from_numpy(scale_factor).to(det_bboxes[0].device)
                for scale_factor in scale_factors
            ]
        _bboxes = [
            det_bboxes[i][:, :4] * scale_factors[i]
            if rescale
            else det_bboxes[i][:, :4]
            for i in range(len(det_bboxes))
        ]

        # Forward
        mask_rois = bbox2roi(_bboxes)
        num_mask_rois_per_img = tuple(_bbox.size(0) for _bbox in _bboxes)
        aug_masks, aug_mask_ious = [], []
        for stage in range(self.num_stages):
            # Mask forward
            mask_roi_extractor = self.mask_roi_extractor[stage]
            mask_head = self.mask_head[stage]

            mask_feats = mask_roi_extractor(x[: mask_roi_extractor.num_inputs], mask_rois)
            mask_pred = mask_head(mask_feats)

            # Scoring
            concat_det_labels = torch.cat(det_labels)
            mask_iou_pred = self.mask_iou_head(
                mask_feats,
                mask_pred[range(concat_det_labels.size(0)), concat_det_labels]
            )

            # split batch mask prediction back to each image
            mask_pred = mask_pred.split(num_mask_rois_per_img, 0)
            mask_iou_preds = mask_iou_pred.split(num_mask_rois_per_img, 0)
            aug_masks.append(
                [m.sigmoid().cpu().detach().numpy() for m in mask_pred]
            )
            aug_mask_ious.append(mask_iou_preds)

        # Apply post-processing to each image individually
        segm_results, mask_scores = [], []
        for i in range(num_imgs):
            if det_bboxes[i].shape[0] == 0:  # No dets
                segm_results.append(
                    [[] for _ in range(self.mask_head[-1].num_classes)]
                )
                mask_scores.append(
                        [[] for _ in range(self.mask_head.num_classes)]
                )
            else:
                # Masks
                aug_mask = [mask[i] for mask in aug_masks]
                merged_masks = merge_aug_masks(
                    aug_mask, [[img_metas[i]]] * self.num_stages, rcnn_test_cfg
                )
                segm_result = self.mask_head[-1].get_seg_masks(
                    merged_masks,
                    _bboxes[i],
                    det_labels[i],
                    rcnn_test_cfg,
                    ori_shapes[i],
                    scale_factors[i],
                    rescale,
                )
                segm_results.append(segm_result)

                # Mask scores
                aug_mask_iou = [aug_mask_iou[i] for aug_mask_iou in aug_mask_ious]
                aug_mask_iou = torch.stack(aug_mask_iou).mean(0)

                mask_score = self.mask_iou_head.get_mask_scores(
                    aug_mask_iou, det_bboxes[i], det_labels[i]
                )
                mask_scores.append(mask_score)

            # Concat mask scores to bbox
            for c in range(len(bbox_results[i])):
                bbox_results[i][c] = np.concatenate(
                    [bbox_results[i][c], np.array(mask_score[c])[:, None]], -1
                )

        return list(zip(bbox_results, segm_results))
