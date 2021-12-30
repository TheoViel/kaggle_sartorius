import cv2
import torch
import numpy as np
import pycocotools

from tqdm.notebook import tqdm


def efficient_concat(arrays):
    """
    Efficient concatenate function.
    Adapted to the case where one array is much longer than the others.
    Thhis function changes the order by concatenating by decreasing length.

    Args:
        arrays (list of np arrays): Arrays to concatenate.

    Returns:
        np array: Concatenated array.
        np array: Order the arrays were concatenated in.
    """
    lens = np.array([len(m) for m in arrays])
    order = np.argsort(lens)[::-1]
    lens = lens[order]

    array = arrays[order[0]]
    end = array.shape[0]

    pad_len = lens[1:].sum()
    if not pad_len:
        return array, order

    array = np.pad(array, ((0, pad_len), (0, 0), (0, 0)))

    for idx, length in zip(order[1:], lens[1:]):
        array[end: end + length] = arrays[idx]
        end += length

    return array, order


def quick_post_process_preds(result, thresh_conf=0.5, thresh_mask=0.5, num_classes=3):
    """
    Quick post-processing function for MMDet results.

    Args:
        result (tuple): MMDet results (boxes, masks).
        thresh_conf (float, optional): Confidence threshold. Defaults to 0.5.
        thresh_mask (float, optional): Mask threshold. Defaults to 0.5.
        num_classes (int, optional): Number of classes. Defaults to 3.

    Returns:
        np array [n x H x W]: Masks.
        np array [n x 5]: Boxes & confidences.
        int: Cell type index.
    """
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
    """
    Removes the overlap between cells.
    The cell i has its intersection with the all the k < i cells removed.

    Args:
        masks (np array [n x H x W]): Masks.
        ious (np array, optional): Precomputed ious between cells. Defaults to None.

    Returns:
        np array [n x H x W]: Processed masks.
    """
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


def remove_overlap_(masks, boxes, ious=None):
    """
    Removes the overlap between cells.

    Args:
        masks (np array [n x H x W]): Masks.
        ious (np array, optional): Precomputed ious between cells. Defaults to None.

    Returns:
        np array [n x H x W]: Processed masks.
    """
    order = np.argsort(masks.sum(-1).sum(-1))
    masks, boxes = masks[order], boxes[order]

    if ious is None:
        rles = [pycocotools.mask.encode(np.asarray(m, order='F')) for m in masks]
        ious = pycocotools.mask.iou(rles, rles, [0] * len(rles))
    else:
        ious = ious[order]
        ious = ious.T[order].T

    for i in range(len(ious)):
        ious[i, i] = 0

    to_process = np.where(ious.sum(0) > 0)[0]

    if not len(to_process):
        return masks, boxes

    masks = torch.from_numpy(masks).cuda()

    for idx, i in enumerate(to_process):
        if idx == 0:
            continue

        indices = [j for j in np.where(ious[i] > 0)[0] if j < i]
        if len(indices):
            others = masks[indices].max(0)[0]
            masks[i] *= ~others

    masks = masks.cpu().numpy()

    # assert masks.sum(0).max() == 1

    return masks, boxes


def mask_nms(masks, boxes, threshold=0.5):
    """
    Non-maximum suppression with masks.

    Args:
        masks (np array [n x H x W]): Masks.
        boxes (np array [n x 5]): Boxes & confidences.
        threshold (float, optional): IoU threshold. Defaults to 0.5.

    Returns:
        np array [m x H x W]: Kept masks.
        np array [m x 5]: Kept boxes.
        list [m]: Kept indices.
    """
    order = np.argsort(boxes[:, 4])[::-1]
    masks = masks[order]
    boxes = boxes[order]

    rle_pred = [pycocotools.mask.encode(np.asarray(m, order='F')) for m in masks]
    ious = pycocotools.mask.iou(rle_pred, rle_pred, [0] * len(rle_pred))

    picks = []
    idxs = list(range(len(ious)))

    while len(idxs) > 0:
        idx = idxs[0]
        overlapping = np.where(ious[idx] > threshold)[0]

        if len(overlapping):
            picks.append(idx)
            idxs = [i for i in idxs if i not in overlapping]
        else:
            idxs = idxs[1:]

    masks = masks[picks]
    boxes = boxes[picks]
    return masks, boxes, picks


def corrupt_mask(mask, draw_contours=False):
    """
    Corrupts a mask in the same fashion the annotations are corrupted :
    mask -> contour -> fill poly.

    Args:
        mask (np array [H x W]): Mask.
        draw_contours (bool, optional): Whether to draw contours for viz. Defaults to False.

    Returns:
        np array [H x W]: Corrupted mask.
        np array [H x W] or None: Drawn contours.
    """
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    if draw_contours:
        img_contours = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
        img_contours = cv2.drawContours(img_contours, contours, -1, (255, 255, 255), 1)[:, :, 0]
    else:
        img_contours = None

    corrupted_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)

    for c in contours:
        corrupted_mask = cv2.fillConvexPoly(corrupted_mask, points=c, color=(1, 1, 1))
    corrupted_mask = corrupted_mask[:, :, 0].astype(mask.dtype)

    return corrupted_mask, img_contours


def remove_small_masks(masks, boxes, min_size=0):
    """
    TODO

    Args:
        masks ([type]): [description]
        boxes ([type]): [description]
        min_size (int, optional): [description]. Defaults to 0.

    Returns:
        [type]: [description]
    """
    if min_size == 0:
        return masks, boxes

    sizes = masks.sum(-1).sum(-1)
    to_keep = sizes > min_size

    if to_keep.min() == 1:
        return masks, boxes

    smallest = sizes.min()
    to_keep = sizes > smallest

    return masks[to_keep], boxes[to_keep]


def process_results(
    results,
    thresholds_mask,
    thresholds_nms,
    thresholds_conf,
    min_sizes,
    remove_overlap=True,
    corrupt=True,
):
    """
    Complete results processing function.
    TODO

    Args:
        results (list of tuples [n]): Results in the MMDet format [(boxes, masks), ...].
        thresholds_mask (list of float [3]): Thresholds per class for masks.
        thresholds_nms (list of float [3]): Thresholds per class for nms.
        thresholds_conf (list of float [3]): Thresholds per class for confidence.
        remove_overlap (bool, optional): Whether to remove overlap. Defaults to True.
        corrupt (bool, optional): Whether to corrupt astro masks. Defaults to True.

    Returns:
        list of np arrays [n]: Masks.
        list of np arrays [n]: Boxes.
        list of ints [n]: Cell types.
    """
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
        min_size = (
            min_sizes if isinstance(min_sizes, (float, int))
            else min_sizes[cell]
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

        # Remove small masks
        masks, boxes = remove_small_masks(masks, boxes, min_size=min_size)

        # masks, boxes = morphology_pp(masks, boxes, cell)

        # Corrupt
        if corrupt and cell == 1:  # astro
            masks = np.array([corrupt_mask(mask)[0] for mask in masks])

        # Remove overlap
        if remove_overlap:
            masks = remove_overlap_naive(masks)
            # masks, boxes = remove_overlap_(masks, boxes)

        all_masks.append(masks)
        all_boxes.append(boxes)

    return all_masks, all_boxes, cell_types
