import torch

from mmdet.core import bbox2roi
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

    def __init__(self, mask_iou_head, **kwargs):
        assert mask_iou_head is not None
        super(CascadeMaskScoringRoIHead, self).__init__(**kwargs)
        self.mask_iou_head = build_head(mask_iou_head)

    def _mask_forward_train(self, x, sampling_results, bbox_feats, gt_masks, img_metas):
        """
        Run forward function and calculate loss for Mask head in training.
        """
        # Get Cascade output
        mask_results = super(CascadeMaskScoringRoIHead, self)._mask_forward_train(
            x, sampling_results, bbox_feats, gt_masks, img_metas
        )
        if mask_results["loss_mask"] is None:
            return mask_results

        # mask iou head forward and loss
        # TODO : Mask scoring head on the last pred ? On the average ? On all preds ? use several heads ?  # noqa

        # Get predictions for positive masks
        pos_labels = torch.cat([res.pos_gt_labels for res in sampling_results])

        pos_mask_pred = mask_results["mask_pred"][
            range(mask_results["mask_pred"].size(0)), pos_labels
        ]
        mask_iou_pred = self.mask_iou_head(mask_results["mask_feats"], pos_mask_pred)
        pos_mask_iou_pred = mask_iou_pred[range(mask_iou_pred.size(0)), pos_labels]

        # Get targets
        mask_iou_targets = self.mask_iou_head.get_targets(
            sampling_results,
            gt_masks,
            pos_mask_pred,
            mask_results["mask_targets"],
            self.train_cfg,
        )

        # Compute losses
        loss_mask_iou = self.mask_iou_head.loss(pos_mask_iou_pred, mask_iou_targets)
        mask_results["loss_mask"].update(loss_mask_iou)

        # TODO : Make sure loss is included : Need to modify the CascadeMaskRCNN class ?

        return mask_results

    def simple_test_mask(self, x, img_metas, det_bboxes, det_labels, rescale=False):
        """
        Obtain mask prediction without augmentation.
        """
        # image shapes of images in the batch
        ori_shapes = tuple(meta["ori_shape"] for meta in img_metas)
        scale_factors = tuple(meta["scale_factor"] for meta in img_metas)
        num_imgs = len(det_bboxes)

        # No predictions
        if all(det_bbox.shape[0] == 0 for det_bbox in det_bboxes):  # No
            num_classes = self.mask_head.num_classes
            segm_results = [[[] for _ in range(num_classes)] for _ in range(num_imgs)]
            mask_scores = [[[] for _ in range(num_classes)] for _ in range(num_imgs)]

            return list(zip(segm_results, mask_scores))

        # Rescale boxes
        if rescale and not isinstance(scale_factors[0], float):
            scale_factors = [
                torch.from_numpy(scale_factor).to(det_bboxes[0].device)
                for scale_factor in scale_factors
            ]
        _bboxes = [
            det_bboxes[i][:, :4] * scale_factors[i] if rescale else det_bboxes[i]
            for i in range(num_imgs)
        ]

        # Get masks
        mask_rois = bbox2roi(_bboxes)
        mask_results = self._mask_forward(x, mask_rois)

        # Score masks
        mask_feats = mask_results["mask_feats"]
        mask_pred = mask_results["mask_pred"]
        concat_det_labels = torch.cat(det_labels)

        mask_iou_pred = self.mask_iou_head(
            mask_feats,
            mask_pred[range(concat_det_labels.size(0)), concat_det_labels],
        )
        # Split batch mask prediction back to each image
        num_bboxes_per_img = tuple(len(_bbox) for _bbox in _bboxes)
        mask_preds = mask_pred.split(num_bboxes_per_img, 0)
        mask_iou_preds = mask_iou_pred.split(num_bboxes_per_img, 0)

        # Apply mask post-processing to each image individually
        segm_results = []
        mask_scores = []
        for i in range(num_imgs):
            if det_bboxes[i].shape[0] == 0:
                segm_results.append([[] for _ in range(self.mask_head.num_classes)])
                mask_scores.append([[] for _ in range(self.mask_head.num_classes)])
            else:
                segm_result = self.mask_head.get_seg_masks(
                    mask_preds[i],
                    _bboxes[i],
                    det_labels[i],
                    self.test_cfg,
                    ori_shapes[i],
                    scale_factors[i],
                    rescale,
                )
                # get mask scores with mask iou head
                mask_score = self.mask_iou_head.get_mask_scores(
                    mask_iou_preds[i], det_bboxes[i], det_labels[i]
                )
                segm_results.append(segm_result)
                mask_scores.append(mask_score)

        return list(zip(segm_results, mask_scores))
