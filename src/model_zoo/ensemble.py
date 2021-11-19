# https://github.com/boliu61/open-images-2019-instance-segmentation/blob/52a7ec2c254deb7b702aa7a085855e31a5254624/mmdetection/mmdet/models/detectors/ensemble_model.py#L7

import mmcv
import torch
from torch import nn
from mmdet.core import bbox_mapping
from mmdet.core import bbox2roi, merge_aug_masks
from mmdet.models.detectors import BaseDetector

from model_zoo.merging import merge_aug_proposals, merge_aug_bboxes, single_class_boxes_nms


""" Faster but not memory efficient ft extraction

all_features = [model.extract_feats(imgs) for model in self.models]

Replace

for model in self.models:
    for x, img_meta in zip(model.extract_feats(imgs), img_metas):

With

for model, features in zip(self.models, all_features):
    for x, img_meta in zip(features, img_metas):
"""


class EnsembleModel(BaseDetector):
    def __init__(self, models, use_tta=False):
        super().__init__()
        self.models = nn.ModuleList([model.module for model in models])
        self.n_models = len(self.models) * 4 if use_tta else len(self.models)

        self.use_tta_prosals = False

        self.rpn_cfg = mmcv.Config(
            dict(
                score_thr=0.,
                max_per_img=None,
                nms=dict(type="nms", iou_threshold=0.7),   # adjust depending on len(models)
                min_bbox_size=0,
            )
        )
        self.rcnn_cfg = mmcv.Config(
            dict(
                score_thr=0.2,
                nms=dict(type="nms", iou_threshold=0.7),
                max_per_img=None,
                mask_thr_binary=-1,
            )
        )

        self.num_classes = self.models[0].roi_head.bbox_head.num_classes
        self.update_model_configs()

    def update_model_configs(self):
        nms_pre = 1000 if (self.n_models > 4 and self.use_tta_prosals) else 2000
        max_per_img = 500 if (self.n_models > 4 and self.use_tta_prosals) else 1000

        nms_pre = 2000
        max_per_img = 1000

        for model in self.models:
            model.rpn_head.test_cfg = mmcv.Config(
                dict(
                    nms_pre=nms_pre,
                    max_per_img=max_per_img,
                    nms=dict(type="nms", iou_threshold=0.7),
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

        for model in self.models:
            for x, img_meta in zip(model.extract_feats(imgs), img_metas):
                aug_img_metas.append(img_meta)

                proposal_list = model.rpn_head.simple_test_rpn(x, img_meta)
                for i, proposals in enumerate(proposal_list):
                    aug_proposals[i].append(proposals)
                    aug_img_metas[i] += img_meta

                if not self.use_tta_prosals:
                    break

        proposal_list = [
            merge_aug_proposals(proposals, img_meta, self.rpn_cfg)
            for proposals, img_meta in zip(aug_proposals, aug_img_metas)
        ]

        return proposal_list, aug_proposals

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

        for model in self.models:
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

        for model in self.models:
            for x, img_meta in zip(model.extract_feats(imgs[:2]), img_metas[:2]):
                img_shape = img_meta[0]["img_shape"]
                scale_factor = img_meta[0]["scale_factor"]
                flip = img_meta[0]["flip"]
                flip_direction = img_meta[0]["flip_direction"]

                _bboxes = bbox_mapping(
                    det_bboxes[:, :4], img_shape, scale_factor, flip, flip_direction
                )
                mask_rois = bbox2roi([_bboxes])
                mask_results = model.roi_head._mask_forward(x, mask_rois)

                aug_masks.append(mask_results["mask_pred"].sigmoid().cpu().numpy())
                aug_img_metas.append(img_meta)

        merged_masks = merge_aug_masks(aug_masks, aug_img_metas, None)

        ori_shape = img_metas[0][0]["ori_shape"]

        masks = self.models[0].roi_head.mask_head.get_seg_masks(
            merged_masks,
            det_bboxes,
            det_labels,
            self.rcnn_cfg,
            ori_shape,
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
            imgs, img_metas, bboxes[:, :5], bboxes[:, -1].long()
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
