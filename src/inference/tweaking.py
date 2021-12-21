import pycocotools
import numpy as np
from tqdm.notebook import tqdm

from utils.metrics import iou_map
from inference.post_process import remove_overlap_naive, corrupt_mask


def mask_nms_multithresh(masks, boxes, thresholds=[0.5], ious=None):
    """
    Non-maximum suppression with masks at multiple thresholds.

    Args:
        masks (np array [n x H x W]): Masks.
        boxes (np array [n x 5]): Boxes & confidences.
        threshold (list of floats, optional): IoU thresholds. Defaults to [0.5].
        ious  (np array, optional): Precomputed ious. Defaults to None.

    Returns:
        list of lists: Kept indices at different thresholds.
    """

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


def evaluate_at_confidences(masks, boxes, confidences, rle_truth):
    """
    Evaluates the mAP IoU at different confidences.

    Args:
        masks (list of np arrays): Predicted masks
        boxes (list of np arrays): Predicted boxes & confidences.
        confidences (list of floats): Confidence thresholds.
        rle_truth (list): ground truths

    Returns:
        list: Scores
    """
    # order = np.argsort(boxes[:, 4])[::-1]
    # masks = masks[order]
    # boxes = boxes[order]
    # assert confidences == sorted(confidences)

    lasts = []
    for thresh in confidences:
        last = np.argmax(boxes[:, 4] < thresh) if np.min(boxes[:, 4]) < thresh else len(boxes)
        lasts.append(last)

    rle_pred = [pycocotools.mask.encode(np.asarray(p, order='F')) for p in masks]

    iou = pycocotools.mask.iou(rle_truth.tolist(), rle_pred, [0] * len(rle_pred))

    scores = [iou_map(ious=[iou[:, :last]]) for last in lasts]

    return scores


def tweak_thresholds(
    results,
    dataset,
    thresholds_mask,
    thresholds_nms,
    thresholds_conf,
    remove_overlap=True,
    corrupt=True,
    num_classes=3,
):
    """
    Function to tweak thresholds for masks, nms and confidence.

    Args:
        results (list of tuples): Results in the MMDet format [(boxes, masks), ...].
        dataset (SartoriusDataset): Dataset containing ground truths.
        thresholds_mask (list of floats): Mask thresholds.
        thresholds_nms (list of floats): NMS thresholds
        thresholds_conf (list of floats): Confidence thresholds.
        remove_overlap (bool, optional): Whether to remove overlap.. Defaults to True.
        corrupt (bool, optional): Whether to corrupt astro cells. Defaults to True.
        num_classes (int, optional): Number of classes. Defaults to 3.

    Returns:
        list of np arrays [3 x n_th_mask x n_th_nms x n_th_conf]: Scores per class for each config.
        list of ints [len(dataset)]: Cell types.
    """
    scores = [[[[] for _ in thresholds_nms] for _ in thresholds_mask] for _ in range(num_classes)]
    cell_types = []

    for idx_mask, threshold_mask in enumerate(thresholds_mask):
        for idx, (result, rle_truth) in tqdm(
            enumerate(zip(results, dataset.encodings)), total=len(dataset)
        ):
            boxes, masks = result

            cell_type_pred = np.argmax(np.bincount(boxes[:, 5].astype(int)))
            cell_type = np.argmax(np.bincount(boxes[:, 5].astype(int)))

            if idx_mask == 0:
                cell_types.append(cell_type)

            masks = masks > (threshold_mask * 255)

            order = np.argsort(boxes[:, 4])[::-1]
            masks = masks[order]
            boxes = boxes[order]

            # Precompute IoUs to save time
            rles = [pycocotools.mask.encode(np.asarray(m, order='F')) for m in masks]
            ious = pycocotools.mask.iou(rles, rles, [0] * len(rles))

            # NMS
            picks = mask_nms_multithresh(masks, boxes, thresholds_nms, ious=ious)

            # Evaluation for different confidences
            for idx_nms, pick in enumerate(picks):
                masks_picked = masks[pick]

                if corrupt and cell_type_pred == 1:  # astro
                    masks_picked = np.array([corrupt_mask(mask)[0] for mask in masks_picked])

                if remove_overlap:
                    masks_picked = remove_overlap_naive(masks_picked, ious=ious[pick].T[pick])

                score = evaluate_at_confidences(
                    masks_picked,
                    boxes[pick],
                    thresholds_conf,
                    rle_truth,
                )
                scores[cell_type][idx_mask][idx_nms].append(score)

    scores = [np.array(s) for s in scores]

    return scores, cell_types
