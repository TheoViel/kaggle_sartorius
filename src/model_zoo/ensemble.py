# https://github.com/boliu61/open-images-2019-instance-segmentation/blob/52a7ec2c254deb7b702aa7a085855e31a5254624/mmdetection/mmdet/models/detectors/ensemble_model.py#L7

import mmcv
import torch
from torch import nn
from mmdet.core import bbox_mapping
from mmdet.core import bbox2roi, merge_aug_masks
from mmdet.models.detectors import BaseDetector

from model_zoo.merging import (
    merge_aug_proposals,
    merge_aug_bboxes,
    single_class_boxes_nms,
)


class EnsembleModel(BaseDetector):
    def __init__(
        self,
        models,
        use_tta=False,
        names=[],
        use_tta_proposals=False,
        single_fold_proposals=False
    ):
        super().__init__()
        self.models = nn.ModuleList([model.module for model in models])
        self.names = names

        self.num_classes = 3
        self.bbox_iou_threshold = 0.7

        self.use_tta_proposals = use_tta_proposals
        self.single_fold_proposals = single_fold_proposals

        self.rpn_cfg = mmcv.Config(
            dict(
                score_thr=0.0,
                max_per_img=None,
                nms=dict(
                    type="nms", iou_threshold=self.bbox_iou_threshold
                ),
                min_bbox_size=0,
            )
        )
        self.rcnn_cfg = mmcv.Config(
            dict(
                score_thr=0.25,
                nms=dict(type="nms", iou_threshold=self.bbox_iou_threshold),
                max_per_img=None,
                mask_thr_binary=-1,
            )
        )

        self.update_model_configs()

    def update_model_configs(self):
        n_models = (
            len([n for n in self.names if n.endswith('_0.pt')])
            if self.single_fold_proposals
            else len(self.models)
        )
        n_fwd = n_models + 3 * n_models * self.use_tta_proposals

        nms_pre = 1000 if n_fwd > 5 else 2000
        max_per_img = 500 if n_fwd > 5 else 1000

        # nms_pre = 1500
        # max_per_img = 750

        for model in self.models:
            model.rpn_head.test_cfg = mmcv.Config(
                dict(
                    nms_pre=nms_pre,
                    max_per_img=max_per_img,
                    nms=dict(type="nms", iou_threshold=self.bbox_iou_threshold),
                    min_bbox_size=0,
                )
            )
            model.roi_head.test_cfg = mmcv.Config({})

    def extract_feat(self, img, img_metas, **kwargs):
        pass

    def simple_test(self, img, img_metas, **kwargs):
        pass

    def forward(self, img, img_metas, **kwargs):
        return self.aug_test(img, img_metas, **kwargs)

    def get_proposals(self, imgs, img_metas):
        """
        TODO
        No TTAs are used.
        TTAs can be used but nms_pre and max_per_img need to be lowered.

        Args:
            imgs ([type]): [description]
            img_metas ([type]): [description]

        Returns:
            [type]: [description]
        """
        imgs_per_gpu = len(img_metas[0])
        aug_proposals = [[] for _ in range(imgs_per_gpu)]
        aug_img_metas = [[] for _ in range(imgs_per_gpu)]

        for i, model in enumerate(self.models):

            if self.single_fold_proposals:
                if not self.names[i].endswith('_0.pt'):
                    continue

            for x, img_meta in zip(model.extract_feats(imgs), img_metas):
                aug_img_metas.append(img_meta)

                proposal_list = model.rpn_head.simple_test_rpn(x, img_meta)
                for i, proposals in enumerate(proposal_list):
                    aug_proposals[i].append(proposals)
                    aug_img_metas[i] += img_meta

                if not self.use_tta_proposals:
                    break

        proposal_list = [
            merge_aug_proposals(proposals, img_meta, self.rpn_cfg)
            for proposals, img_meta in zip(aug_proposals, aug_img_metas)
        ]

        return proposal_list, aug_proposals

    @staticmethod
    def get_boxes_rcnn(model, x, rois, img_shape, scale_factor, num_classes):
        bbox_results = model.roi_head._bbox_forward(x, rois)
        bboxes, scores = model.roi_head.bbox_head.get_bboxes(
            rois,
            bbox_results["cls_score"],
            bbox_results["bbox_pred"],
            img_shape,
            scale_factor,
            rescale=False,
            cfg=None,
        )

        # Keep only desired classes
        scores = scores[:, :num_classes]

        # Keep box corresponding to most confident class
        _, det_labels = torch.max(scores, 1)

        bboxes = bboxes.view(bboxes.size(0), -1, 4)
        bboxes = torch.stack([bboxes[i, c] for i, c in enumerate(det_labels)])

        return bboxes, scores

    @staticmethod
    def get_boxes_cascade(model, x, rois, img_shape, scale_factor, img_meta, num_classes):
        # https://github.com/open-mmlab/mmdetection/blob/bde7b4b7eea9dd6ee91a486c6996b2d68662366d/mmdet/models/roi_heads/test_mixins.py#L139

        ms_scores = []
        for i in range(model.roi_head.num_stages):
            bbox_results = model.roi_head._bbox_forward(i, x, rois)
            ms_scores.append(bbox_results["cls_score"])

            if i < model.roi_head.num_stages - 1:
                cls_score = bbox_results["cls_score"]
                if model.roi_head.bbox_head[i].custom_activation:
                    cls_score = model.roi_head.bbox_head[i].loss_cls.get_activation(
                        cls_score
                    )
                bbox_label = cls_score[:, :-1].argmax(dim=1)
                rois = model.roi_head.bbox_head[i].regress_by_class(
                    rois, bbox_label, bbox_results["bbox_pred"], img_meta[0]
                )

        cls_score = sum(ms_scores) / float(len(ms_scores))
        bboxes, scores = model.roi_head.bbox_head[-1].get_bboxes(
            rois,
            cls_score,
            bbox_results["bbox_pred"],
            img_shape,
            scale_factor,
            rescale=False,
            cfg=None,
        )

        scores = scores[:, :num_classes]

        return bboxes, scores

    def get_bboxes(self, imgs, img_metas, proposal_list):
        """
        https://github.com/open-mmlab/mmdetection/blob/bde7b4b7eea9dd6ee91a486c6996b2d68662366d/mmdet/models/roi_heads/test_mixins.py#L139
        TODO
        All TTAs are used.

        Args:
            imgs ([type]): [description]
            img_metas ([type]): [description]
            proposal_list ([type]): [description]

        Returns:
            [type]: [description]
        """
        aug_bboxes, aug_scores, aug_img_metas = [], [], []

        for name, model in zip(self.names, self.models):
            for x, img_meta in zip(model.extract_feats(imgs), img_metas):
                img_shape = img_meta[0]["img_shape"]
                scale_factor = img_meta[0]["scale_factor"]
                flip = img_meta[0]["flip"]
                flip_direction = img_meta[0]["flip_direction"]

                proposals = bbox_mapping(
                    proposal_list[0][:, :4],
                    img_shape,
                    scale_factor,
                    flip,
                    flip_direction,
                )
                rois = bbox2roi([proposals])

                # Mask RCNN
                if "cascade" in name:
                    bboxes, scores = self.get_boxes_cascade(
                        model, x, rois, img_shape, scale_factor, img_meta, self.num_classes
                    )
                elif "rcnn" in name:
                    bboxes, scores = self.get_boxes_rcnn(
                        model, x, rois, img_shape, scale_factor, self.num_classes
                    )
                else:
                    raise NotImplementedError

                aug_bboxes.append(bboxes)
                aug_scores.append(scores)
                aug_img_metas.append(img_meta)

        merged_bboxes, merged_scores = merge_aug_bboxes(
            aug_bboxes, aug_scores, aug_img_metas, self.rcnn_cfg
        )

        det_bboxes, det_labels = single_class_boxes_nms(
            merged_bboxes,
            merged_scores,
            iou_threshold=self.rcnn_cfg.nms.iou_threshold,
        )

        det_bboxes = torch.cat([det_bboxes, det_labels.unsqueeze(-1)], -1)

        det_bboxes = det_bboxes[det_bboxes[:, 4] > self.rcnn_cfg.score_thr]

        return det_bboxes, merged_bboxes, aug_bboxes

    def get_masks(self, imgs, img_metas, det_bboxes, det_labels):
        """
        https://github.com/open-mmlab/mmdetection/blob/bde7b4b7eea9dd6ee91a486c6996b2d68662366d/mmdet/models/roi_heads/test_mixins.py#L282
        TODO

        Only hflip TTA is used.

        Args:
            imgs ([type]): [description]
            img_metas ([type]): [description]
            det_bboxes ([type]): [description]
            det_labels ([type]): [description]

        Returns:
            [type]: [description]
        """
        aug_masks, aug_img_metas = [], []

        for name, model in zip(self.names, self.models):
            for x, img_meta in zip(model.extract_feats(imgs[:2]), img_metas[:2]):
                img_shape = img_meta[0]["img_shape"]
                scale_factor = img_meta[0]["scale_factor"]
                flip = img_meta[0]["flip"]
                flip_direction = img_meta[0]["flip_direction"]

                _bboxes = bbox_mapping(
                    det_bboxes[:, :4], img_shape, scale_factor, flip, flip_direction
                )
                mask_rois = bbox2roi([_bboxes])

                if "cascade" in name:
                    masks = []
                    for i in range(model.roi_head.num_stages):
                        mask = model.roi_head._mask_forward(i, x, mask_rois)['mask_pred']
                        mask = mask.sigmoid()[:, :self.num_classes]
                        masks.append(mask)
                    mask = torch.stack(masks)
                    mask = mask.mean(0).cpu().numpy()

                elif "rcnn" in name:
                    mask = model.roi_head._mask_forward(x, mask_rois)['mask_pred']
                    mask = mask.sigmoid().cpu().numpy()[:, :self.num_classes]

                else:
                    raise NotImplementedError

                aug_masks.append(mask)
                aug_img_metas.append(img_meta)

        merged_masks = merge_aug_masks(aug_masks, aug_img_metas, None)

        mask_head = (
            self.models[0].roi_head.mask_head[-1] if "cascade" in self.names[0]
            else self.models[0].roi_head.mask_head
        )

        masks = mask_head.get_seg_masks(
            merged_masks,
            det_bboxes,
            det_labels,
            self.rcnn_cfg,
            img_metas[0][0]["ori_shape"],
            scale_factor=det_bboxes.new_ones(4),
            rescale=False,
            return_per_class=False,
        )

        return masks, merged_masks, aug_masks

    def aug_test(self, imgs, img_metas, return_everything=False, **kwargs):
        """
        Adapted from :
        https://github.com/open-mmlab/mmdetection/blob/bde7b4b7eea9dd6ee91a486c6996b2d68662366d/mmdet/models/roi_heads/standard_roi_head.py#L268
        """

        proposal_list, aug_proposals = self.get_proposals(imgs, img_metas)

        bboxes, merged_bboxes, aug_bboxes = self.get_bboxes(
            imgs, img_metas, proposal_list
        )

        assert self.models[0].with_mask

        if bboxes.shape[0] == 0:
            return bboxes, None

        masks, merged_masks, aug_masks = self.get_masks(
            imgs, img_metas, bboxes[:, :5], bboxes[:, 5].long()
        )

        if return_everything:
            return (bboxes, masks), (
                proposal_list,
                aug_proposals,
                bboxes,
                merged_bboxes,
                aug_bboxes,
                masks,
                merged_masks,
                aug_masks,
            )

        return (bboxes, masks)
