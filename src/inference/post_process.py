import cv2
import torch
import numpy as np
import pycocotools

from tqdm.notebook import tqdm


def efficient_concat(masks, lens=None):
    if lens is None:
        lens = np.array([len(m) for m in masks])
    order = np.argsort(lens)[::-1]
    lens = lens[order]

    mask = masks[order[0]]
    end = mask.shape[0]

    pad_len = lens[1:].sum()
    if not pad_len:
        return mask, order

    mask = np.pad(mask, ((0, pad_len), (0, 0), (0, 0)))

    for idx, length in zip(order[1:], lens[1:]):
        mask[end: end + length] = masks[idx]
        end += length

    return mask, order


def quick_post_process_preds(result, thresh_conf=0.5, thresh_mask=0.5, num_classes=3):
    masks, boxes = [], []

    lens = [len(boxes_c) for boxes_c, masks_c in zip(result[0], result[1])][:num_classes]
    cell = np.argmax(lens)

    # Get masks & filter by confidence
    for c, (boxes_c, masks_c) in enumerate(zip(result[0], result[1])):
        scores = boxes_c[:, -1]  # uses iou score for mask_scoring !

        if len(scores):
            last = np.argmax(scores < thresh_conf) if np.min(scores) < thresh_conf else len(masks_c)
            if last > 0:
                masks.append(np.array(masks_c[:last]) > (thresh_mask * 255))
                boxes.append(boxes_c[:last])

        if c == num_classes - 1:
            break

    if not len(masks):
        return [], [], cell

    masks, order = efficient_concat(masks)
    boxes = np.concatenate(np.array(boxes)[order])

    return masks, boxes, cell


def remove_overlap_naive(masks, ious=None):
    if ious is None:
        rles = [pycocotools.mask.encode(np.asarray(m, order='F')) for m in masks]
        ious = pycocotools.mask.iou(rles, rles, [0] * len(rles))

    for i in range(len(ious)):
        ious[i, i] = 0

    to_process = np.where(ious.sum(0) > 0)[0]

    if not len(to_process):
        return masks

    masks = torch.from_numpy(masks).cuda()
    overlapping_masks = masks[to_process]

    for idx, i in enumerate(to_process):
        if idx == 0:
            continue
        others = overlapping_masks[:idx].max(0)[0]
        masks[i] *= ~others

    return masks.cpu().numpy()


def mask_nms(masks, boxes, threshold=0.5):
    """
    NMS with masks.
    Removes more masks than the tweaking fct.

    Args:
        masks ([type]): [description]
        boxes ([type]): [description]
        threshold (float, optional): [description]. Defaults to 0.5.

    Returns:
        [type]: [description]
    """
    # assert list(np.argsort(boxes[:, 4])[::-1]) == list(range(len(boxes)))

    order = np.argsort(boxes[:, 4])[::-1]
    masks = masks[order]
    boxes = boxes[order]

    rle_pred = [pycocotools.mask.encode(np.asarray(m, order='F')) for m in masks]
    ious = pycocotools.mask.iou(rle_pred, rle_pred, [0] * len(rle_pred))

    picks = []
    idxs = list(range(len(ious)))
    # removed = []

    while len(idxs) > 0:
        idx = idxs[0]
        overlapping = np.where(ious[idx] > threshold)[0]

        # removed += [v for v in overlapping if v > idx]

        if len(overlapping):
            picks.append(idx)
            idxs = [i for i in idxs if i not in overlapping]
        else:
            idxs = idxs[1:]

    masks = masks[picks]
    boxes = boxes[picks]
    return masks, boxes, picks


def mask_nms_multithresh(masks, boxes, thresholds=[0.5], ious=None):
    assert thresholds == sorted(thresholds)

    if thresholds == [0]:
        return [range(len(boxes))]

    # Compute ious
    if ious is None:
        rle_pred = [pycocotools.mask.encode(np.asarray(m, order='F')) for m in masks]
        ious = pycocotools.mask.iou(rle_pred, rle_pred, [0] * len(rle_pred))

    all_idxs = [list(range(len(ious))) for _ in range(len(thresholds))]
    picks = [[] for _ in thresholds]

    # NMS
    for idx in range(len(ious)):
        if not any([idx in all_idx for all_idx in all_idxs]):
            continue

        overlappings = [np.where(ious[idx] > t)[0] for t in thresholds]  # overlaps for current idx

        for i, overlapping in enumerate(overlappings):  # update masks to remove
            if idx in all_idxs[i]:
                if len(overlapping):
                    picks[i].append(idx)
                    all_idxs[i] = [j for j in all_idxs[i] if j not in overlapping]
                else:
                    all_idxs[i] = all_idxs[i][1:]

    return picks


def process_results(
    results, thresholds_mask, thresholds_nms, thresholds_conf, remove_overlap=True, corrupt=False
):
    all_masks, all_boxes, cell_types = [], [], []

    for result in tqdm(results):
        boxes, masks = result

        # Cell type
        cell = np.argmax(np.bincount(boxes[:, 5].astype(int)))
        cell_types.append(cell)

        # Thresholds
        thresh_mask = (
            thresholds_mask if isinstance(thresholds_mask, (float, int))
            else thresholds_mask[cell]
        )
        thresh_nms = (
            thresholds_nms if isinstance(thresholds_nms, (float, int))
            else thresholds_nms[cell]
        )
        thresh_conf = (
            thresholds_conf if isinstance(thresholds_conf, (float, int))
            else thresholds_conf[cell]
        )

        # Binarize
        masks = masks > (thresh_mask * 255)

        # Sort by decreasing conf
        order = np.argsort(boxes[:, 4])[::-1]
        masks = masks[order]
        boxes = boxes[order]

        # Remove low confidence
        last = (
            np.argmax(boxes[:, 4] < thresh_conf) if np.min(boxes[:, 4]) < thresh_conf
            else len(boxes)
        )
        masks = masks[:last]
        boxes = boxes[:last]

        # NMS
        if thresh_nms > 0:
            masks, boxes, _ = mask_nms(masks, boxes, thresh_nms)

        # Remove overlap
        if remove_overlap:
            masks = remove_overlap_naive(masks)

        # Corrupt
        if corrupt and cell == 1:  # astro
            masks = np.array([degrade_mask(mask) for mask in masks])

        all_masks.append(masks)
        all_boxes.append(boxes)

    return all_masks, all_boxes, cell_types


def degrade_mask(mask):
    cont, hier = cv2.findContours(mask.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    img_cont = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    img_cont = cv2.drawContours(img_cont, cont, -1, (255, 255, 255), 1)
    img_cont = img_cont[:, :, 0]

    conv_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)

    for c in cont:
        conv_mask = cv2.fillConvexPoly(conv_mask, points=c, color=(255, 255, 255))
    conv_mask = (conv_mask[:, :, 0] > 0).astype(np.uint8)

    return conv_mask, img_cont
