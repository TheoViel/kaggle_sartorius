import skimage
import pycocotools
import numpy as np

from inference.post_process import quick_post_process_preds


def dice_score(pred, truth, eps=1e-8, threshold=0.5):
    """
    Dice metric.
    Args:
        pred (np array): Predictions.
        truth (np array): Ground truths.
        eps (float, optional): epsilon to avoid dividing by 0. Defaults to 1e-8.
        threshold (float, optional): Threshold for predictions. Defaults to 0.5.
    Returns:
        float: dice value.
    """
    pred = (pred.reshape((truth.shape[0], -1)) > threshold).astype(int)
    truth = truth.reshape((truth.shape[0], -1)).astype(int)
    intersect = (pred + truth == 2).sum(-1)
    union = pred.sum(-1) + truth.sum(-1)
    dice = (2.0 * intersect + eps) / (union + eps)
    return dice.mean()


def bbox_iou(bb1, bb2):
    """
    IoU for two bounding boxes in the format (x0, x1, y0, y1).

    Args:
        bb1 (list [4]): 1st box coordinates.
        bb2 (list [4]): 2nd box coordinates.

    Returns:
        float: IoU.
    """
    # determine the coordinates of the intersection rectangle
    x_left = max(bb1[0], bb2[0])
    y_top = max(bb1[1], bb2[1])
    x_right = min(bb1[2], bb2[2])
    y_bottom = min(bb1[3], bb2[3])

    if x_right < x_left or y_bottom < y_top:
        return 0.

    # The intersection of two axis-aligned bounding boxes is always an
    # axis-aligned bounding box
    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # compute the area of both AABBs
    bb1_area = (bb1[2] - bb1[0]) * (bb1[3] - bb1[1])
    bb2_area = (bb2[2] - bb2[0]) * (bb2[3] - bb2[1])

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = intersection_area / float(bb1_area + bb2_area - intersection_area)

    return iou


def compute_iou(labels, y_pred):
    """
    Computes the IoU for instance labels and predictions.

    Args:
        labels (np array): Labels.
        y_pred (np array): predictions

    Returns:
        np array: IoU matrix, of size true_objects x pred_objects.
    """

    true_objects = len(np.unique(labels))
    pred_objects = len(np.unique(y_pred))

    if true_objects != labels.max() + 1:
        labels, _, _ = skimage.segmentation.relabel_sequential(labels)

    if pred_objects != y_pred.max() + 1:
        y_pred, _, _ = skimage.segmentation.relabel_sequential(y_pred)

    # Compute intersection between all objects
    intersection = np.histogram2d(
        labels.flatten(), y_pred.flatten(), bins=(true_objects, pred_objects)
    )[0]

    # Compute areas (needed for finding the union between all objects)
    area_true = np.histogram(labels, bins=true_objects)[0]
    area_pred = np.histogram(y_pred, bins=pred_objects)[0]
    area_true = np.expand_dims(area_true, -1)
    area_pred = np.expand_dims(area_pred, 0)

    # Compute union
    union = area_true + area_pred - intersection

    iou = intersection / (union + 1e-6)
    return iou[1:, 1:]


def precision_at(threshold, iou):
    """
    Computes the precision at a given threshold.

    Args:
        threshold (float): Threshold.
        iou (np array [n_truths x n_preds]): IoU matrix.

    Returns:
        int: Number of true positives,
        int: Number of false positives,
        int: Number of false negatives.
    """
    matches = iou > threshold
    # true_positives = np.sum(matches, axis=0) >= 1  # Correct objects
    true_positives = (np.sum(matches, axis=1) == 1).sum()  # Correct objects
    false_negatives = (np.sum(matches, axis=1) == 0).sum()
    false_positives = (np.sum(matches, axis=0) == 0).sum() + (np.sum(matches, axis=1) > 1).sum()

    return true_positives, false_positives, false_negatives


def iou_map(truths=None, preds=None, ious=None, verbose=0):
    """
    Computes the metric for the competition.
    Masks contain the segmented pixels where each object has one value associated,
    and 0 is the background.
    IoUs can be precomputed.

    Args:
        truths (list of masks, optional): Ground truths.
        preds (list of masks, optional): Predictions.
        ious (list of matrices, optional): Precomputed ious. Of size n_truths x n_preds.
        verbose (int, optional): Whether to print infos. Defaults to 0.

    Returns:
        float: mAP.
    """
    if preds is None or truths is None:
        assert ious is not None

    if ious is None:
        assert truths is not None and preds is not None
        ious = [compute_iou(truth, pred) for truth, pred in zip(truths, preds)]

    if verbose:
        print("Thresh\tTP\tFP\tFN\tPrec.")

    all_precs = []
    for t in np.arange(0.5, 1.0, 0.05):
        tps, fps, fns = 0, 0, 0
        prec = []
        for iou in ious:
            tp, fp, fn = precision_at(t, iou)
            tps += tp
            fps += fp
            fns += fn

            p = tps / (tps + fps + fns) if tps else 0
            prec.append(p)

        all_p = np.mean(prec)
        all_precs.append(p)
        if verbose:
            print("{:1.3f}\t{}\t{}\t{}\t{:1.3f}".format(t, tps, fps, fns, all_p))

    if verbose:
        print("AP\t-\t-\t-\t{:1.3f}".format(np.mean(all_precs)))

    return np.mean(all_precs)


def quick_eval_results(dataset, results, num_classes=3):
    """
    Evaluate predictions directly from the mmdet output.
    Performs post processing with the default parameters.

    Args:
        dataset (SartoriusDataset): Dataset containing ground truths.
        results (list of tuples): Results in the MMDet format [(boxes, masks), ...].
        num_classes (int, optional): Number of classes. Defaults to 3.

    Returns:
        float: Overall score
        list [num_classes]: Scores per class.
    """
    precs = [[] for _ in range(num_classes)]
    for idx in range(len(dataset)):
        masks, _, cell_type = quick_post_process_preds(
            results[idx], num_classes=num_classes,
        )

        if not len(masks):
            precs[cell_type].append(0)
            continue

        rle_pred = [pycocotools.mask.encode(np.asarray(p, order='F')) for p in masks]
        rle_truth = dataset.encodings[idx].tolist()

        iou = pycocotools.mask.iou(rle_truth, rle_pred, [0] * len(rle_pred))
        score = iou_map(ious=[iou])
        precs[cell_type].append(score)

    return np.mean(np.concatenate(precs)), [np.mean(p) for p in precs if len(p)]


def evaluate(masks_pred, dataset, cell_types):
    """
    Evaluate predictions.

    Args:
        masks_pred (list of masks [n x nb_cell x h x w]): Predicted masks.
        dataset (SartoriusDataset): Dataset containing ground truths.
        cell_types (list of ints): Predicted cell types.

    Returns:
        list: Scores per image.
        list of 3 lists: Scores per image per class.
    """
    scores = []
    scores_per_class = [[], [], []]

    for masks, cell_type, rle_truth in zip(masks_pred, cell_types, dataset.encodings):
        rle_pred = [pycocotools.mask.encode(np.asarray(p, order='F')) for p in masks]

        iou = pycocotools.mask.iou(rle_truth.tolist(), rle_pred, [0] * len(rle_pred))
        score = iou_map(ious=[iou])

        scores_per_class[cell_type].append(score)
        scores.append(score)

    return scores, scores_per_class
