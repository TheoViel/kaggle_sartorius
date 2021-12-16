import mmcv
import torch
import numpy as np

from torch import nn
from mmcv.ops import nms
from mmdet.core import bbox_mapping
from mmdet.core import bbox2roi, merge_aug_masks
from mmdet.models.detectors import BaseDetector

from model_zoo.wrappers import get_wrappers
from model_zoo.custom_head_functions import get_rpn_boxes, get_seg_masks
from model_zoo.merging import merge_aug_bboxes, single_class_boxes_nms


DELTA = 0.5  # Modify this to change the TTA shift


class AutomatedShiftFix(nn.Module):
    def __init__(self, min_shift_x=0, max_shift_x=0, min_shift_y=0, max_shift_y=0):
        super().__init__()
        self.shifts_x = np.arange(min_shift_x, max_shift_x + 1)
        self.shifts_y = np.arange(min_shift_y, max_shift_y + 1)

    @staticmethod
    def compute_similarities(x, xs):
        return (xs * x.unsqueeze(0)).sum((1, 2, 3))

    @staticmethod
    def shift(x, shifts):
        if shifts == (0, 0):
            return x

        shifted = torch.roll(x, shifts=shifts, dims=(1, 2))

        shift_x, shift_y = shifts
        if shift_x > 0:
            shifted[:, :shift_x] = 0
        elif shift_x < 0:
            shifted[:, shift_x:] = 0
        if shift_y > 0:
            shifted[:, :, :shift_y] = 0
        elif shift_y < 0:
            shifted[:, :, shift_y:] = 0

        return shifted

    def build_shifted(self, x):
        xs, shifts = [], []
        for shift_x in self.shifts_x:
            for shift_y in self.shifts_y:
                shifted = self.shift(x, shifts=(shift_x, shift_y))
                xs.append(shifted)
                shifts.append((shift_x, shift_y))
        xs = torch.stack(xs)
        return xs, shifts

    def forward(self, x_ref, x):
        xs, shifts = [], []
        for i in range(x.size(0)):

            xs_, shifts_ = self.build_shifted(x[i])
            sims = self.compute_similarities(x_ref[i], xs_)
            best = sims.argmax()

            xs.append(xs_[best].cpu().numpy())
            shifts.append(shifts_[best])

        return np.array(xs), shifts


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

        self.shifters = {
            "horizontal": AutomatedShiftFix(max_shift_y=1),
            "vertical": AutomatedShiftFix(max_shift_x=1),
            "diagonal": AutomatedShiftFix(max_shift_x=1, max_shift_y=1),
        }

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
        Extract features function. Features are stored in cpu to save gpu memory.

        Args:
            imgs (list of torch tensors [n x C x H x W]): Input image.
            img_metas (list of dicts [n]): List of MMDet image metadata.
        """
        features = [
            [tuple([f_lvl.cpu() for f_lvl in f]) for f in model.extract_feats(img)]
            for model in self.models
        ]
        return features

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
        Forward function. Calls the aug_test function.

        Args:
            imgs (list of torch tensors [n_tta x C x H x W]): Input image.
            img_metas (list of dicts [n_tta]): List of MMDet image metadata.

        Returns:
            torch tensor [m x 6]: Kept boxes, confidences & labels.
            torch tensor [m x H x W]: Masks.
            list of torch tensors [1 x 5]: Proposals.
            list of torch tensors: Augmented boxes before merging.
            list of torch tensors: Augmented masks before merging.
        """
        return self.aug_test(img, img_metas, **kwargs)

    def get_proposals(self, features, img_metas):
        """
        Gets proposals. Doesn't use TTA.

        Args:
            features (list of torch tensors [n_models x n_tta x n_ft]): Encoder / FPN features.
            img_metas (list of dicts [n_tta]): List of MMDet image metadata.

        Returns:
            list of torch tensors [1 x 5]: Proposals.
            int: Cell type.
        """
        aug_bboxes, aug_scores = [], []
        for i, model in enumerate(self.models):
            for fts, img_meta in zip(features[i][:1], img_metas[:1]):
                fts = [ft.cuda() for ft in fts]
                cls_score, bbox_pred = model.rpn_head(fts)

                aug_bboxes.append(bbox_pred)
                aug_scores.append(cls_score)

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

    def get_proposals_tta(self, features, img_metas):
        """
        Gets proposals. Uses TTA.
        This works worse than without TTA.

        Args:
            features (list of torch tensors [n_models x n_tta x n_ft]): Encoder / FPN features.
            img_metas (list of dicts [n_tta]): List of MMDet image metadata.

        Returns:
            list of torch tensors [1 x 5]: Proposals.
            int: Cell type.
        """

        aug_bboxes, aug_scores, aug_img_metas = {}, {}, {}

        for i, model in enumerate(self.models):
            for fts, img_meta in zip(features[i], img_metas):
                fts = [ft.cuda() for ft in fts]

                flip_direction = str(img_meta[0]["flip_direction"])

                cls_score, bbox_pred = model.rpn_head(fts)

                try:
                    aug_bboxes[flip_direction].append(bbox_pred)
                    aug_scores[flip_direction].append(cls_score)
                    aug_img_metas[flip_direction].append(img_meta)
                except KeyError:
                    aug_bboxes[flip_direction] = [bbox_pred]
                    aug_scores[flip_direction] = [cls_score]
                    aug_img_metas[flip_direction] = [img_meta]

        all_proposals, cell_types = [], []
        for flip_direction in aug_bboxes.keys():
            merged_bboxes, merged_scores, level_counts = [], [], []

            for lvl in range(len(aug_bboxes[flip_direction][0])):
                merged_bboxes_lvl = torch.stack(
                    [bboxes[lvl] for bboxes in aug_bboxes[flip_direction]]
                ).mean(dim=0)
                merged_scores_lvl = torch.stack(
                    [scores[lvl] for scores in aug_scores[flip_direction]]
                ).mean(dim=0)

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

            img_meta = aug_img_metas[flip_direction][0]
            proposals = get_rpn_boxes(
                self.models[0].rpn_head,
                merged_scores,
                merged_bboxes,
                img_meta,
                self.rpn_cfgs[cell_type]
            )[0]

            proposals[:, :4] = bbox_mapping(
                proposals[:, :4],
                img_meta[0]["img_shape"],
                img_meta[0]["scale_factor"],
                img_meta[0]["flip"],
                img_meta[0]["flip_direction"],
            )

            all_proposals.append(proposals)
            cell_types.append(cell_type)

        proposals = torch.cat(all_proposals, 0)
        cell_type = np.argmax(np.bincount(cell_types))

        proposals, _ = nms(
            proposals[:, :4].contiguous(),
            proposals[:, 4].contiguous(),
            0.9
        )
        # proposals = torch.cat([all_proposals[0], proposals])

        # proposals = torch.unique(proposals, dim=0)
        # _, order = proposals[:, 4].sort(0, descending=True)
        # proposals = proposals[order]

        # proposals = proposals[:self.rpn_cfgs[cell_type].max_per_img]

        return [proposals], cell_type

    def get_bboxes(self, features, img_metas, proposal_list, rcnn_cfg):
        """
        Gets rcnn boxes. Adapted from :
        https://github.com/open-mmlab/mmdetection/blob/bde7b4b7eea9dd6ee91a486c6996b2d68662366d/mmdet/models/roi_heads/test_mixins.py#L139
        All TTAs are used.

        Args:
            features (list of torch tensors [n_models x n_tta x n_ft]): Encoder / FPN features.
            img_metas (list of dicts [n_tta]): List of MMDet image metadata.
            proposal_list ([1 x N]): Proposals.
            rcnn_cfg (mmdet Config): RCNN config.

        Returns:
            torch tensor [m x 6]: Kept boxes, confidences & labels.
            list of torch tensors: Augmented boxes before merging.
        """
        aug_bboxes, aug_scores, aug_img_metas = [], [], []

        for i, (wrapper, model) in enumerate(zip(self.wrappers, self.models)):
            for fts, img_meta in zip(features[i], img_metas):
                fts = [ft.cuda() for ft in fts]
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

                # # Not that useful ?
                # if flip_direction in ['vertical', 'diagonal']:
                #     rois[:, 2] = torch.clamp(rois[:, 2] - DELTA, 0, img_shape[0])
                #     rois[:, 4] = torch.clamp(rois[:, 4] - DELTA, 0, img_shape[0])
                # if flip_direction in ['horizontal', 'diagonal']:
                #     rois[:, 1] = torch.clamp(rois[:, 1] - DELTA, 0, img_shape[1])
                #     rois[:, 3] = torch.clamp(rois[:, 3] - DELTA, 0, img_shape[1])

                bboxes, scores = wrapper.get_boxes(
                    model, fts, rois, img_shape, scale_factor, img_meta, self.config['num_classes']
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

    def get_masks(self, features, img_metas, det_bboxes, det_labels):
        """
        Gets rcnn boxes. Adapted from :
        https://github.com/open-mmlab/mmdetection/blob/bde7b4b7eea9dd6ee91a486c6996b2d68662366d/mmdet/models/roi_heads/test_mixins.py#L282

        Only hflip TTA is used.

        Args:
            features (list of torch tensors [n_models x n_tta x n_ft]): Encoder / FPN features.
            img_metas (list of dicts [n_tta]): List of MMDet image metadata.
            det_bboxes (torch tensor [m x 5]): Boxes & confidences.
            det_labels (torch tensor [m]): Labels.

        Returns:
            torch tensor [m x H x W]: Masks.
            list of torch tensors: Augmented masks before merging.
        """
        aug_masks, aug_img_metas = [], []

        for i, (wrapper, model) in enumerate(zip(self.wrappers, self.models)):
            for fts, img_meta in zip(features[i][:2], img_metas[:2]):
                # for fts, img_meta in zip(features[i], img_metas):
                fts = [ft.cuda() for ft in fts]
                img_shape = img_meta[0]["img_shape"]
                scale_factor = img_meta[0]["scale_factor"]
                flip = img_meta[0]["flip"]
                flip_direction = img_meta[0]["flip_direction"]

                _bboxes = bbox_mapping(
                    det_bboxes[:, :4], img_shape, scale_factor, flip, flip_direction
                )
                mask_rois = bbox2roi([_bboxes])

                # # Seems to help
                # if flip_direction in ['vertical', 'diagonal']:
                #     mask_rois[:, 2] = torch.clamp(mask_rois[:, 2] - DELTA, 0, img_shape[0])
                #     mask_rois[:, 4] = torch.clamp(mask_rois[:, 4] - DELTA, 0, img_shape[0])
                # if flip_direction in ['horizontal', 'diagonal']:
                #     mask_rois[:, 1] = torch.clamp(mask_rois[:, 1] - DELTA, 0, img_shape[1])
                #     mask_rois[:, 3] = torch.clamp(mask_rois[:, 3] - DELTA, 0, img_shape[1])

                masks = wrapper.get_masks(model, fts, mask_rois, self.config['num_classes'])

                # if flip_direction is None:
                #     masks_ref = torch.from_numpy(masks).cuda()
                # else:
                #     masks, shifts = self.shifters[flip_direction](
                #         masks_ref, torch.from_numpy(masks).cuda()
                #     )

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
        features = self.extract_feat(imgs, img_metas)

        proposal_list, cell_type = self.get_proposals(features, img_metas)
        # proposal_list, cell_type = self.get_proposals_tta(features, img_metas)

        bboxes, aug_bboxes = self.get_bboxes(
            features, img_metas, proposal_list, self.rcnn_cfgs[cell_type]
        )

        assert self.models[0].with_mask

        if bboxes.shape[0] == 0:
            return bboxes, None

        masks, aug_masks = self.get_masks(
            features, img_metas, bboxes[:, :5], bboxes[:, 5].long()
        )

        if return_everything:
            all_stuff = (proposal_list, aug_bboxes, bboxes, aug_masks, masks)
            return (bboxes, masks), all_stuff

        return (bboxes, masks)
