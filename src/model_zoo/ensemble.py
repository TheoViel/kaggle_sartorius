# https://github.com/boliu61/open-images-2019-instance-segmentation/blob/52a7ec2c254deb7b702aa7a085855e31a5254624/mmdetection/mmdet/models/detectors/ensemble_model.py#L7

from torch import nn
from mmdet.core import bbox2result, bbox_mapping
from mmdet.core import (
    bbox2roi,
    merge_aug_masks,
    merge_aug_bboxes,
    multiclass_nms,
    merge_aug_proposals,
)
from mmdet.models.detectors import BaseDetector


class EnsembleModel(BaseDetector):
    def __init__(self, models):
        super().__init__()
        self.models = nn.ModuleList([model.module for model in models])

        self.rpn_test_cfg = self.models[0].test_cfg.rpn
        self.rcnn_test_cfg = self.models[0].test_cfg.rcnn
        self.num_classes = self.models[0].roi_head.bbox_head.num_classes

    def simple_test(self, img, img_meta, **kwargs):
        pass

    def forward_train(self, imgs, img_metas, **kwargs):
        pass

    def extract_feat(self, imgs):
        pass

    def forward(self, img, img_metas, **kwargs):
        return self.aug_test(img, img_metas, **kwargs)

    def aug_test(self, imgs, img_metas, **kwargs):
        """
        Test with augmentations.
        If rescale is False, then returned bboxes and masks will fit the scale
        of imgs[0].
        Adapted from https://github.com/open-mmlab/mmdetection/blob/bde7b4b7eea9dd6ee91a486c6996b2d68662366d/mmdet/models/roi_heads/standard_roi_head.py#L268  # noqa
        """
        imgs_per_gpu = len(img_metas[0])
        aug_proposals = [[] for _ in range(imgs_per_gpu)]

        # Extract features
        # This is done 3 times to save memory but it's faster to compute once if memory allows it.
        """
        all_features = [model.extract_feats(imgs) for model in self.models]

        Replace

        for model in self.models:
            for x, img_meta in zip(model.extract_feats(imgs), img_metas):

        With

        for model, features in zip(self.models, all_features):
            for x, img_meta in zip(features, img_metas):
        """

        for model in self.models:
            for x, img_meta in zip(model.extract_feats(imgs), img_metas):
                proposal_list = model.rpn_head.simple_test_rpn(x, img_meta)
                for i, proposals in enumerate(proposal_list):
                    aug_proposals[i].append(proposals)

        proposal_list = [
            merge_aug_proposals(proposals, img_meta, self.rpn_test_cfg)
            for proposals, img_meta in zip(aug_proposals, img_metas)
        ]

        # Extract bboxes
        # https://github.com/open-mmlab/mmdetection/blob/bde7b4b7eea9dd6ee91a486c6996b2d68662366d/mmdet/models/roi_heads/test_mixins.py#L139
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
            aug_bboxes, aug_scores, aug_img_metas, self.rcnn_test_cfg
        )
        det_bboxes, det_labels = multiclass_nms(
            merged_bboxes,
            merged_scores,
            self.rcnn_test_cfg.score_thr,
            self.rcnn_test_cfg.nms,
            self.rcnn_test_cfg.max_per_img,
        )

        bbox_result = bbox2result(det_bboxes, det_labels, self.num_classes)

        # Extract masks
        # https://github.com/open-mmlab/mmdetection/blob/bde7b4b7eea9dd6ee91a486c6996b2d68662366d/mmdet/models/roi_heads/test_mixins.py#L282
        assert self.models[0].with_mask

        if det_bboxes.shape[0] == 0:
            segm_result = [[] for _ in range(self.num_classes - 1)]
            return bbox_result, segm_result

        aug_masks, aug_img_metas = [], []

        for model in self.models:
            for x, img_meta in zip(model.extract_feats(imgs), img_metas):
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

        merged_masks = merge_aug_masks(aug_masks, aug_img_metas, self.rcnn_test_cfg)

        ori_shape = img_metas[0][0]["ori_shape"]
        segm_result = self.models[0].roi_head.mask_head.get_seg_masks(
            merged_masks,
            det_bboxes,
            det_labels,
            self.rcnn_test_cfg,
            ori_shape,
            scale_factor=det_bboxes.new_ones(4),
            rescale=False,
        )
        return [[bbox_result, segm_result]]
