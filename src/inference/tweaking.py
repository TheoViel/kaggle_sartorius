import pycocotools
import numpy as np
from tqdm.notebook import tqdm

from utils.metrics import iou_map
from inference.post_process import remove_overlap_naive, mask_nms_multithresh


def evaluate_at_confidences(masks, boxes, confidences, rle_truth):
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
    results, dataset, thresholds_mask, thresholds_nms, thresholds_conf, remove_overlap=False
):
    scores = [[[[] for _ in thresholds_nms] for _ in thresholds_mask] for _ in range(3)]

    for idx_mask, threshold_mask in enumerate(thresholds_mask):
        for result, rle_truth in tqdm(zip(results, dataset.encodings), total=len(dataset)):
            boxes, masks = result

            cell_type = np.argmax(np.bincount(boxes[:, 5].astype(int)))

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

    return scores
