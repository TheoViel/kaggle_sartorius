import mmcv
import torch
import numpy as np

from torch import nn
from mmdet.core import bbox_mapping
from mmdet.core import bbox2roi, merge_aug_masks
from mmdet.models.detectors import BaseDetector

from model_zoo.wrappers import get_wrappers
from model_zoo.custom_head_functions import get_rpn_boxes, get_seg_masks
from model_zoo.merging import merge_aug_bboxes, single_class_boxes_nms


class EnsembleModel(BaseDetector):
    """
    Wrapper to ensemble models.
    """
    def __init__(
        self,
        models,
        config,
        names=[],
    ):
        """
        Constructor.

        Args:
            models (list of mmdet MMDataParallel): Models to ensemble.
            config (dict): Ensemble config.
            names (list, optional): Model names. Defaults to [].
        """
        super().__init__()
        self.models = nn.ModuleList([model.module for model in models])
        self.config = config
        self.names = names

        self.wrappers = get_wrappers(self.names)
        self.get_configs()

    def get_configs(self):
        """
        Creates the rpn and rcnn configs from the config dict.
        """
        self.rpn_cfgs, self.rcnn_cfgs = [], []

        for i in range(3):
            rpn_cfg = mmcv.Config(
                dict(
                    score_thr=self.config['rpn_score_threshold'][i],
                    nms_pre=self.config['rpn_nms_pre'][i],
                    max_per_img=self.config['rpn_max_per_img'][i],
                    nms=dict(type="nms", iou_threshold=self.config['rpn_iou_threshold'][i]),
                    min_bbox_size=0,
                )
            )
            rcnn_cfg = mmcv.Config(
                dict(
                    score_thr=self.config['rcnn_score_threshold'][i],
                    nms=dict(type="nms", iou_threshold=self.config['rcnn_iou_threshold'][i]),
                    mask_thr_binary=-1,
                )
            )
            self.rpn_cfgs.append(rpn_cfg)
            self.rcnn_cfgs.append(rcnn_cfg)

    def extract_feat(self, img, img_metas, **kwargs):
        """
        Extract features function. Not used but required by MMDet.

        Args:
            imgs (list of torch tensors [n x C x H x W]): Input image.
            img_metas (list of dicts [n]): List of MMDet image metadata.
        """
        pass

    def simple_test(self, img, img_metas, **kwargs):
        """
        Single image test function. Not used but required by MMDet.

        Args:
            imgs (list of torch tensors [1 x C x H x W]): Input image.
            img_metas (list of dicts [1]): List of MMDet image metadata.
        """
        pass

    def forward(self, img, img_metas, **kwargs):
        """
        Forward function.

        Args:
            imgs (list of torch tensors [n_tta x C x H x W]): Input image.
            img_metas (list of dicts [n_tta]): List of MMDet image metadata.

        Returns:
            [type]: [description]
        """
        return self.aug_test(img, img_metas, **kwargs)

    def get_proposals(self, imgs, img_metas):
        """
        Gets proposals, doesn't use TTA.

        Args:
            imgs (list of torch tensors [n_tta x C x H x W]): Input image.
            img_metas (list of dicts [n_tta]): List of MMDet image metadata.

        Returns:
            list of torch tensors [1 x 5]: Proposals.
            int: Cell type.
        """
        aug_bboxes, aug_scores = [], []
        for i, model in enumerate(self.models):
            for x, img_meta in zip(model.extract_feats(imgs), img_metas):
                cls_score, bbox_pred = model.rpn_head(x)

                aug_bboxes.append(bbox_pred)
                aug_scores.append(cls_score)

                break  # no tta

        merged_bboxes, merged_scores = [], []
        level_counts = []

        for lvl in range(len(aug_bboxes[0])):
            merged_bboxes_lvl = torch.stack([bboxes[lvl] for bboxes in aug_bboxes]).mean(dim=0)
            merged_scores_lvl = torch.stack([scores[lvl] for scores in aug_scores]).mean(dim=0)

            merged_bboxes.append(merged_bboxes_lvl)
            merged_scores.append(merged_scores_lvl)

            rpn_scores_lvl, rpn_labels_lvl = torch.max(
                merged_scores_lvl.sigmoid().flatten(start_dim=2)[0], 0
            )
            level_counts.append(rpn_labels_lvl[rpn_scores_lvl > 0.7].size(0))

        if np.sum(level_counts[-2:]) > 10:  # astro
            cell_type = 1
        elif np.sum(level_counts) < 4500 and level_counts[1] < 750:  # cort
            cell_type = 2
        else:  # shsy5y
            cell_type = 0

        proposal_list = get_rpn_boxes(
            self.models[0].rpn_head,
            merged_scores,
            merged_bboxes,
            img_metas[0],
            self.rpn_cfgs[cell_type]
        )

        return proposal_list, cell_type

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
        raise NotImplementedError

    def get_bboxes(self, imgs, img_metas, proposal_list, rcnn_cfg):
        """
        Gets rcnn boxes. Adapted from :
        https://github.com/open-mmlab/mmdetection/blob/bde7b4b7eea9dd6ee91a486c6996b2d68662366d/mmdet/models/roi_heads/test_mixins.py#L139
        All TTAs are used.

        Args:
            imgs (list of torch tensors [n_tta x C x H x W]): Input images.
            img_metas (list of dicts [n_tta]): List of MMDet image metadata.
            proposal_list ([1 x N]): Proposals.

        Returns:
            torch tensor [m x 6]: Kept boxes, confidences & labels.
            list of torch tensors: Augmented boxes before merging.
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
                    model, x, rois, img_shape, scale_factor, img_meta, self.config['num_classes']
                )

                aug_bboxes.append(bboxes)
                aug_scores.append(scores)
                aug_img_metas.append(img_meta)

        merged_bboxes, merged_scores = merge_aug_bboxes(
            aug_bboxes, aug_scores, aug_img_metas
        )

        if self.config['bbox_nms']:
            det_bboxes, det_labels = single_class_boxes_nms(
                merged_bboxes,
                merged_scores,
                iou_threshold=rcnn_cfg.nms.iou_threshold,
            )
            det_bboxes = torch.cat([det_bboxes, det_labels.unsqueeze(-1)], -1)

        else:
            det_scores, det_labels = torch.max(merged_scores, 1)
            det_bboxes = torch.cat(
                [merged_bboxes, det_scores.unsqueeze(1), det_labels.unsqueeze(1)], 1
            )

            _, order = det_scores.sort(0, descending=True)
            det_bboxes = det_bboxes[order]

        det_bboxes = det_bboxes[det_bboxes[:, 4] > rcnn_cfg.score_thr]

        return det_bboxes, torch.cat([merged_bboxes, merged_scores], 1)

    def get_masks(self, imgs, img_metas, det_bboxes, det_labels):
        """
        Gets rcnn boxes. Adapted from :
        https://github.com/open-mmlab/mmdetection/blob/bde7b4b7eea9dd6ee91a486c6996b2d68662366d/mmdet/models/roi_heads/test_mixins.py#L282

        Only hflip TTA is used.

        Args:
            imgs (list of torch tensors [n_tta x C x H x W]): Input images.
            img_metas (list of dicts [n_tta]): List of MMDet image metadata.
            det_bboxes (torch tensor [m x 5): Boxes & confidences.
            det_labels (torch tensor [m]): Labels.

        Returns:
            torch tensor [m x H x W]: Masks.
            list of torch tensors: Augmented masks before merging.
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

                masks = wrapper.get_masks(model, x, mask_rois, self.config['num_classes'])

                aug_masks.append(masks)
                aug_img_metas.append(img_meta)

        merged_masks = merge_aug_masks(aug_masks, aug_img_metas, None)

        mask_head = (
            self.models[0].roi_head.mask_head if "rcnn" in self.names[0]
            else self.models[0].roi_head.mask_head[-1]
        )

        masks = get_seg_masks(
            mask_head,
            merged_masks,
            det_bboxes,
            det_labels,
            self.rcnn_cfgs[0],
            img_metas[0][0]["ori_shape"],
            scale_factor=det_bboxes.new_ones(4),
            rescale=False,
            return_per_class=False,
        )

        return masks, aug_masks

    def aug_test(self, imgs, img_metas, return_everything=False, **kwargs):
        """
        Augmented test function. Adapted from :
        https://github.com/open-mmlab/mmdetection/blob/bde7b4b7eea9dd6ee91a486c6996b2d68662366d/mmdet/models/roi_heads/standard_roi_head.py#L268

        Args:
            imgs (list of torch tensors [n_tta x C x H x W]): Input images.
            img_metas (list of dicts [n_tta]): List of MMDet image metadata.
            return_everything (bool, optional): Whether to return more stuff. Defaults to False.

        Returns:
            torch tensor [m x 6]: Kept boxes, confidences & labels.
            torch tensor [m x H x W]: Masks.
            list of torch tensors [1 x 5]: Proposals.
            list of torch tensors: Augmented boxes before merging.
            list of torch tensors: Augmented masks before merging.
        """
        proposal_list, cell_type = self.get_proposals(imgs, img_metas)

        bboxes, aug_bboxes = self.get_bboxes(
            imgs, img_metas, proposal_list, self.rcnn_cfgs[cell_type]
        )

        assert self.models[0].with_mask

        if bboxes.shape[0] == 0:
            return bboxes, None

        masks, aug_masks = self.get_masks(
            imgs, img_metas, bboxes[:, :5], bboxes[:, 5].long()
        )

        if return_everything:
            all_stuff = (proposal_list, aug_bboxes, bboxes, aug_masks, masks)
            return (bboxes, masks), all_stuff

        return (bboxes, masks)
