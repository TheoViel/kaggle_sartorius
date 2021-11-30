# https://github.com/boliu61/open-images-2019-instance-segmentation/blob/52a7ec2c254deb7b702aa7a085855e31a5254624/mmdetection/mmdet/models/detectors/ensemble_model.py#L7

import mmcv
import torch
from torch import nn
from mmdet.core import bbox_mapping
from mmdet.core import bbox2roi, merge_aug_masks
from mmdet.models.detectors import BaseDetector

from model_zoo.wrappers import get_wrappers
from model_zoo.merging import (  # noqa
    merge_aug_proposals,
    merge_aug_bboxes,
    single_class_boxes_nms,
    merge_aug_proposals
)


class EnsembleModel(BaseDetector):
    def __init__(
        self,
        models,
        names=[],
    ):
        super().__init__()
        self.models = nn.ModuleList([model.module for model in models])
        self.names = names

        self.wrappers = get_wrappers(self.names)

        self.num_classes = 3
        self.bbox_iou_threshold = 0.75

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
        nms_pre = 2000
        max_per_img = 1000

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
        aug_bboxes, aug_scores = [], []
        for i, model in enumerate(self.models):

            for x, img_meta in zip(model.extract_feats(imgs), img_metas):
                cls_score, bbox_pred = model.rpn_head(x)

                aug_bboxes.append(bbox_pred)
                aug_scores.append(cls_score)

                break  # no tta

        merged_bboxes, merged_scores = [], []
        for lvl in range(len(aug_bboxes[0])):
            merged_bboxes_lvl = torch.stack([bboxes[lvl] for bboxes in aug_bboxes]).mean(dim=0)
            merged_scores_lvl = torch.stack([scores[lvl] for scores in aug_scores]).mean(dim=0)

            # print(merged_bboxes_lvl.size(), merged_scores_lvl.size())
            merged_bboxes.append(merged_bboxes_lvl)
            merged_scores.append(merged_scores_lvl)

        proposal_list = self.models[0].rpn_head.get_bboxes(
            merged_scores, merged_bboxes, img_metas[0]
        )

        return proposal_list, (merged_bboxes, merged_scores)

    def get_proposals_tta(self, imgs, img_metas):
        """
        TODO
        This doesn't work yet.

        Args:
            imgs ([type]): [description]
            img_metas ([type]): [description]

        Returns:
            [type]: [description]
        """
        aug_bboxes, aug_scores, aug_img_metas = {}, {}, {}

        for i, model in enumerate(self.models):

            if self.single_fold_proposals:
                if not self.names[i].endswith('_0.pt'):
                    continue

            for x, img_meta in zip(model.extract_feats(imgs), img_metas):
                flip_direction = img_meta[0]["flip_direction"]

                cls_score, bbox_pred = model.rpn_head(x)

                try:
                    aug_bboxes[flip_direction].append(bbox_pred)
                    aug_scores[flip_direction].append(cls_score)
                    aug_img_metas[flip_direction].append(img_meta)
                except KeyError:
                    aug_bboxes[flip_direction] = [bbox_pred]
                    aug_scores[flip_direction] = [cls_score]
                    aug_img_metas[flip_direction] = [img_meta]

        proposals, img_metas_proposals = [], []
        for flip_direction in aug_bboxes.keys():
            merged_bboxes, merged_scores = [], []
            for lvl in range(len(aug_bboxes[flip_direction][0])):
                merged_bboxes_lvl = torch.stack(
                    [bboxes[lvl] for bboxes in aug_bboxes[flip_direction]]
                ).mean(dim=0)
                merged_scores_lvl = torch.stack(
                    [scores[lvl] for scores in aug_scores[flip_direction]]
                ).mean(dim=0)

                # print(merged_bboxes_lvl.size())

                # if flip_direction == "horizontal":
                #     merged_bboxes_lvl = merged_bboxes_lvl.flip([3])
                #     merged_scores_lvl = merged_scores_lvl.flip([3])
                # elif flip_direction == "vertical":
                #     merged_bboxes_lvl = merged_bboxes_lvl.flip([3])
                #     merged_scores_lvl = merged_scores_lvl.flip([3])
                # elif flip_direction == "diagonal":
                #     merged_bboxes_lvl = merged_bboxes_lvl.flip([2, 3])
                #     merged_scores_lvl = merged_scores_lvl.flip([2, 3])

                merged_bboxes.append(merged_bboxes_lvl)
                merged_scores.append(merged_scores_lvl)

            proposal_list = self.models[0].rpn_head.get_bboxes(
                merged_scores, merged_bboxes, aug_img_metas[flip_direction][0]
            )

            if flip_direction in ["horizontal"]:
                # print(len(proposal_list), proposal_list[0].size())
                proposals.append(proposal_list)
                img_metas_proposals.append(aug_img_metas[flip_direction][0])
                # print(aug_img_metas[flip_direction][0])

        proposal_list_final = []

        for i in range(len(proposals[0])):
            # Doesn't work but I could try to use NMS
            merged_bboxes, merged_scores = merge_aug_bboxes(
                [p[i][:, :4] for p in proposals],
                [p[i][:, 4] for p in proposals],
                img_metas_proposals,
            )
            merged_proposal = torch.cat([merged_bboxes, merged_scores.unsqueeze(1)], 1)
            proposal_list_final.append(merged_proposal)

        return proposal_list_final, (merged_bboxes, merged_scores)

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

        for wrapper, model in zip(self.wrappers, self.models):
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

                bboxes, scores = wrapper.get_boxes(
                    model, x, rois, img_shape, scale_factor, img_meta, self.num_classes
                )

                aug_bboxes.append(bboxes)
                aug_scores.append(scores)
                aug_img_metas.append(img_meta)

        merged_bboxes, merged_scores = merge_aug_bboxes(
            aug_bboxes, aug_scores, aug_img_metas
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

        for wrapper, model in zip(self.wrappers, self.models):
            for x, img_meta in zip(model.extract_feats(imgs[:2]), img_metas[:2]):
                img_shape = img_meta[0]["img_shape"]
                scale_factor = img_meta[0]["scale_factor"]
                flip = img_meta[0]["flip"]
                flip_direction = img_meta[0]["flip_direction"]

                _bboxes = bbox_mapping(
                    det_bboxes[:, :4], img_shape, scale_factor, flip, flip_direction
                )
                mask_rois = bbox2roi([_bboxes])

                masks = wrapper.get_masks(model, x, mask_rois, self.num_classes)

                aug_masks.append(masks)
                aug_img_metas.append(img_meta)

        merged_masks = merge_aug_masks(aug_masks, aug_img_metas, None)

        mask_head = (
            self.models[0].roi_head.mask_head if "rcnn" in self.names[0]
            else self.models[0].roi_head.mask_head[-1]
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
