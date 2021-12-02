import torch
from mmcv.ops import nms

from mmdet.core.bbox import bbox_mapping_back


def merge_aug_proposals(aug_proposals, img_metas, cfg):
    """
    Merges proposals.

    Args:
        aug_proposals (list of tensors [k x n x 5]): Proposals from different testing schemes.
        img_metas (list of dicts [k]): List of mmdet image metadata.
        cfg (dict): RPN config.

    Returns:
        tensor [m x 5]: Merged proposals.
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
        aug_bboxes (list of torch tensors [k x n x 4]): Boxes.
        aug_scores (list of torch tensors [k x n]): Confidences.
        img_metas (list of dicts [k]): List of mmdet image metadata.

    Returns:
        torch tensor [n x 4]: Merged boxes.
        torch tensor [n]: Merged scores
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
        return bboxes, None
    else:
        scores = torch.stack(aug_scores).mean(dim=0)
        return bboxes, scores


def single_class_boxes_nms(merged_bboxes, merged_scores, iou_threshold=0.5):
    """
    NMS for boxes but considering all classes at once.

    Args:
        merged_bboxes (torch tensor [n x 4]): Merged boxes.
        merged_scores (torch tensor [n]): Merged scores.
        iou_threshold (float, optional): Threshold of IoU. Defaults to 0.5.

    Returns:
        torch tensor [m x 5]: Kept boxes & confidences.
        torch tensor [m]: Labels of kept boxes.
    """
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
