import copy
import torch
import numpy as np
from warnings import warn

from mmcv.ops import batched_nms
from mmdet.models.roi_heads.mask_heads.fcn_mask_head import (
    BYTES_PER_FLOAT,
    GPU_MEM_LIMIT,
    _do_paste_mask,
)


def get_seg_masks(
    mask_head,
    mask_pred,
    det_bboxes,
    det_labels,
    rcnn_test_cfg,
    ori_shape,
    scale_factor,
    rescale,
    return_per_class=True,
):
    """
    Modified version of mmdet/models/roi_heads/mask_heads/fcn_mask_head.py
    to add the return_per_class argument.
    """
    if isinstance(mask_pred, torch.Tensor):
        mask_pred = mask_pred.sigmoid()
    else:
        # In AugTest, has been activated before
        mask_pred = det_bboxes.new_tensor(mask_pred)

    device = mask_pred.device
    cls_segms = [
        [] for _ in range(mask_head.num_classes)
    ]  # BG is not included in num_classes
    bboxes = det_bboxes[:, :4]
    labels = det_labels

    # In most cases, scale_factor should have been
    # converted to Tensor when rescale the bbox

    if not isinstance(scale_factor, torch.Tensor):
        if isinstance(scale_factor, float):
            scale_factor = np.array([scale_factor] * 4)
            warn(
                "Scale_factor should be a Tensor or ndarray "
                "with shape (4,), float would be deprecated. "
            )
        assert isinstance(scale_factor, np.ndarray)
        scale_factor = torch.Tensor(scale_factor)

    if rescale:
        img_h, img_w = ori_shape[:2]
        bboxes = bboxes / scale_factor.to(bboxes.device)
    else:
        w_scale, h_scale = scale_factor[0], scale_factor[1]
        img_h = np.round(ori_shape[0] * h_scale.item()).astype(np.int32)
        img_w = np.round(ori_shape[1] * w_scale.item()).astype(np.int32)

    N = len(mask_pred)
    if device.type == "cpu":
        num_chunks = N
    else:
        # GPU benefits from parallelism for larger chunks,
        num_chunks = int(
            np.ceil(N * int(img_h) * int(img_w) * BYTES_PER_FLOAT / GPU_MEM_LIMIT)
        )
        assert num_chunks <= N, "Default GPU_MEM_LIMIT is too small; try increasing it"
    chunks = torch.chunk(torch.arange(N, device=device), num_chunks)

    threshold = rcnn_test_cfg.mask_thr_binary
    im_mask = torch.zeros(
        N,
        img_h,
        img_w,
        device=device,
        dtype=torch.bool if threshold >= 0 else torch.uint8,
    )

    if not mask_head.class_agnostic:
        mask_pred = mask_pred[range(N), labels][:, None]

    for inds in chunks:
        masks_chunk, spatial_inds = _do_paste_mask(
            mask_pred[inds], bboxes[inds], img_h, img_w, skip_empty=device.type == "cpu"
        )

        if threshold >= 0:
            masks_chunk = (masks_chunk >= threshold).to(dtype=torch.bool)
        else:
            masks_chunk = (masks_chunk * 255).to(dtype=torch.uint8)

        im_mask[(inds,) + spatial_inds] = masks_chunk

    for i in range(N):
        cls_segms[labels[i]].append(im_mask[i].detach().cpu().numpy())

    if return_per_class:
        return cls_segms
    else:
        return im_mask


def get_rpn_boxes_single(
    rpn_head,
    cls_scores,
    bbox_preds,
    mlvl_anchors,
    img_shape,
    scale_factor,
    cfg,
):
    """
    Modified from mmdet/models/dense_heads/rpn_head.py

    Transform outputs for a single batch item into bbox predictions.

        Args:
        cls_scores (list[Tensor]): Box scores of all scale level
            each item has shape (num_anchors * num_classes, H, W).
        bbox_preds (list[Tensor]): Box energies / deltas of all
            scale level, each item has shape (num_anchors * 4, H, W).
        mlvl_anchors (list[Tensor]): Anchors of all scale level
            each item has shape (num_total_anchors, 4).
        img_shape (tuple[int]): Shape of the input image,
            (height, width, 3).
        scale_factor (ndarray): Scale factor of the image arrange as
            (w_scale, h_scale, w_scale, h_scale).
        cfg (mmcv.Config): Test / postprocessing configuration,
            if None, test_cfg would be used.
    Returns:
        Tensor: Labeled boxes in shape (n, 5), where the first 4 columns
            are bounding box positions (tl_x, tl_y, br_x, br_y) and the
            5-th column is a score between 0 and 1.
    """
    cfg = copy.deepcopy(cfg)
    # bboxes from different level should be independent during NMS,
    # level_ids are used as labels for batched NMS to separate them
    level_ids = []
    mlvl_scores = []
    mlvl_bbox_preds = []
    mlvl_valid_anchors = []
    for idx in range(len(cls_scores)):
        rpn_cls_score = cls_scores[idx]
        rpn_bbox_pred = bbox_preds[idx]
        assert rpn_cls_score.size()[-2:] == rpn_bbox_pred.size()[-2:]
        rpn_cls_score = rpn_cls_score.permute(1, 2, 0)
        if rpn_head.use_sigmoid_cls:
            rpn_cls_score = rpn_cls_score.reshape(-1)
            scores = rpn_cls_score.sigmoid()
        else:
            rpn_cls_score = rpn_cls_score.reshape(-1, 2)
            # We set FG labels to [0, num_class-1] and BG label to
            # num_class in RPN head since mmdet v2.5, which is unified to
            # be consistent with other head since mmdet v2.0. In mmdet v2.0
            # to v2.4 we keep BG label as 0 and FG label as 1 in rpn head.
            scores = rpn_cls_score.softmax(dim=1)[:, 0]
        rpn_bbox_pred = rpn_bbox_pred.permute(1, 2, 0).reshape(-1, 4)
        anchors = mlvl_anchors[idx]
        if cfg.nms_pre > 0 and scores.shape[0] > cfg.nms_pre:
            # sort is faster than topk
            # _, topk_inds = scores.topk(cfg.nms_pre)
            ranked_scores, rank_inds = scores.sort(descending=True)
            topk_inds = rank_inds[: cfg.nms_pre]
            scores = ranked_scores[: cfg.nms_pre]
            rpn_bbox_pred = rpn_bbox_pred[topk_inds, :]
            anchors = anchors[topk_inds, :]
        mlvl_scores.append(scores)
        mlvl_bbox_preds.append(rpn_bbox_pred)
        mlvl_valid_anchors.append(anchors)
        level_ids.append(scores.new_full((scores.size(0),), idx, dtype=torch.long))

    scores = torch.cat(mlvl_scores)
    anchors = torch.cat(mlvl_valid_anchors)
    rpn_bbox_pred = torch.cat(mlvl_bbox_preds)
    proposals = rpn_head.bbox_coder.decode(anchors, rpn_bbox_pred, max_shape=img_shape)
    ids = torch.cat(level_ids)

    if cfg.min_bbox_size >= 0:
        w = proposals[:, 2] - proposals[:, 0]
        h = proposals[:, 3] - proposals[:, 1]
        valid_mask = (w > cfg.min_bbox_size) & (h > cfg.min_bbox_size)
        if not valid_mask.all():
            proposals = proposals[valid_mask]
            scores = scores[valid_mask]
            ids = ids[valid_mask]
    if proposals.numel() > 0:
        dets, keep = batched_nms(proposals, scores, ids, cfg.nms)
    else:
        return proposals.new_zeros(0, 5)

    return dets[: cfg.max_per_img]


def get_rpn_boxes(rpn_head, cls_scores, bbox_preds, img_metas, cfg):
    """
    TODO
    Modified from mmdet/models/dense_heads/rpn_head.py

    Args:
        rpn_head ([type]): [description]
        cls_scores ([type]): [description]
        bbox_preds ([type]): [description]
        img_metas ([type]): [description]
        cfg ([type], optional): [description]. Defaults to None.
        rescale (bool, optional): [description]. Defaults to False.
        with_nms (bool, optional): [description]. Defaults to True.

    Returns:
        [type]: [description]
    """
    assert len(cls_scores) == len(bbox_preds)
    num_levels = len(cls_scores)
    device = cls_scores[0].device
    featmap_sizes = [cls_scores[i].shape[-2:] for i in range(num_levels)]
    mlvl_anchors = rpn_head.anchor_generator.grid_anchors(featmap_sizes, device=device)

    result_list = []
    for img_id in range(len(img_metas)):
        cls_score_list = [cls_scores[i][img_id].detach() for i in range(num_levels)]
        bbox_pred_list = [bbox_preds[i][img_id].detach() for i in range(num_levels)]
        img_shape = img_metas[img_id]["img_shape"]
        scale_factor = img_metas[img_id]["scale_factor"]
        proposals = get_rpn_boxes_single(
            rpn_head,
            cls_score_list,
            bbox_pred_list,
            mlvl_anchors,
            img_shape,
            scale_factor,
            cfg,
        )
        result_list.append(proposals)

    return result_list
