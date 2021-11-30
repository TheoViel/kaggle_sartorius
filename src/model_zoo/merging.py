import torch
from mmcv.ops import nms

from mmdet.core.bbox import bbox_mapping_back


def merge_aug_proposals(aug_proposals, img_metas, cfg):
    """Merge augmented proposals (multiscale, flip, etc.)

    Args:
        aug_proposals (list[Tensor]): proposals from different testing
            schemes, shape (n, 5). Note that they are not rescaled to the
            original image size.

        img_metas (list[dict]): list of image info dict where each dict has:
            'img_shape', 'scale_factor', 'flip', and may also contain
            'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
            For details on the values of these keys see
            `mmdet/datasets/pipelines/formatting.py:Collect`.

        cfg (dict): rpn test config.

    Returns:
        Tensor: shape (n, 4), proposals corresponding to original image scale.
    """
    # Recover augmented proposals
    recovered_proposals = []
    for proposals, img_info in zip(aug_proposals, img_metas):
        img_shape = img_info["img_shape"]
        scale_factor = img_info["scale_factor"]
        flip = img_info["flip"]
        flip_direction = img_info["flip_direction"]
        _proposals = proposals.clone()
        _proposals[:, :4] = bbox_mapping_back(
            _proposals[:, :4], img_shape, scale_factor, flip, flip_direction
        )
        recovered_proposals.append(_proposals)

    # Merge proposals with NMS
    aug_proposals = torch.cat(recovered_proposals, dim=0)
    merged_proposals, _ = nms(
        aug_proposals[:, :4].contiguous(),
        aug_proposals[:, 4].contiguous(),
        cfg.nms.iou_threshold,
    )

    # Reorder
    scores = merged_proposals[:, 4]

    scores, order = scores.sort(0, descending=True)

    order = order[scores > cfg.score_thr]
    order = order[:cfg.max_per_img]

    merged_proposals = merged_proposals[order, :]

    return merged_proposals


def merge_aug_bboxes(aug_bboxes, aug_scores, img_metas):
    """
    Merge augmented detection bboxes and scores.
    This simply takes the mean.

    Args:
        aug_bboxes (list[Tensor]): shape (n, 4*#class)
        aug_scores (list[Tensor] or None): shape (n, #class)
        img_shapes (list[Tensor]): shape (3, ).

    Returns:
        tuple: (bboxes, scores)
    """
    # Recover augmented proposals
    recovered_bboxes = []
    for bboxes, img_info in zip(aug_bboxes, img_metas):
        img_shape = img_info[0]["img_shape"]
        scale_factor = img_info[0]["scale_factor"]
        flip = img_info[0]["flip"]
        flip_direction = img_info[0]["flip_direction"]
        bboxes = bbox_mapping_back(
            bboxes, img_shape, scale_factor, flip, flip_direction
        )
        recovered_bboxes.append(bboxes)

    # Merge boxes by averaging predictions
    bboxes = torch.stack(recovered_bboxes).mean(dim=0)

    if aug_scores is None:
        return bboxes
    else:
        scores = torch.stack(aug_scores).mean(dim=0)
        return bboxes, scores


def single_class_boxes_nms(merged_bboxes, merged_scores, iou_threshold=0.5):
    # Use most confident class per candidate
    det_scores, det_labels = torch.max(merged_scores, 1)

    # Get class & corresponding iou threshold
    cell_type = torch.mode(det_labels, 0).values.item()
    thresh = iou_threshold if isinstance(iou_threshold, (float, int)) else iou_threshold[cell_type]

    # Filter with NMS
    det_bboxes, inds = nms(
        merged_bboxes.contiguous(), det_scores.contiguous(), thresh
    )

    return det_bboxes, det_labels[inds]
