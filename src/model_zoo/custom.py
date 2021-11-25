import numpy as np
import torch

from warnings import warn

from mmdet.models.roi_heads.mask_heads.fcn_mask_head import (
    BYTES_PER_FLOAT,
    GPU_MEM_LIMIT,
    _do_paste_mask,
)


def get_seg_masks(
    self,
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
    cls_segms = [[] for _ in range(self.num_classes)]  # BG is not included in num_classes
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
    # The actual implementation split the input into chunks,
    # and paste them chunk by chunk.
    if device.type == "cpu":
        # CPU is most efficient when they are pasted one by one with
        # skip_empty=True, so that it performs minimal number of
        # operations.
        num_chunks = N
    else:
        # GPU benefits from parallelism for larger chunks,
        # but may have memory issue
        # the types of img_w and img_h are np.int32,
        # when the image resolution is large,
        # the calculation of num_chunks will overflow.
        # so we neet to change the types of img_w and img_h to int.
        # See https://github.com/open-mmlab/mmdetection/pull/5191
        num_chunks = int(np.ceil(N * int(img_h) * int(img_w) * BYTES_PER_FLOAT / GPU_MEM_LIMIT))
        assert num_chunks <= N, "Default GPU_MEM_LIMIT is too small; try increasing it"
    chunks = torch.chunk(torch.arange(N, device=device), num_chunks)

    threshold = rcnn_test_cfg.mask_thr_binary
    im_mask = torch.zeros(
        N, img_h, img_w, device=device, dtype=torch.bool if threshold >= 0 else torch.uint8
    )

    if not self.class_agnostic:
        mask_pred = mask_pred[range(N), labels][:, None]

    for inds in chunks:
        masks_chunk, spatial_inds = _do_paste_mask(
            mask_pred[inds], bboxes[inds], img_h, img_w, skip_empty=device.type == "cpu"
        )

        if threshold >= 0:
            masks_chunk = (masks_chunk >= threshold).to(dtype=torch.bool)
        else:
            # for visualization and debugging
            masks_chunk = (masks_chunk * 255).to(dtype=torch.uint8)

        im_mask[(inds,) + spatial_inds] = masks_chunk

    for i in range(N):
        cls_segms[labels[i]].append(im_mask[i].detach().cpu().numpy())

    if return_per_class:
        return cls_segms
    else:
        return im_mask
