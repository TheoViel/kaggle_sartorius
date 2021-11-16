import skimage
import pycocotools
import numpy as np

from inference.post_process import post_process_preds


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
        iou (np array): IoU matrix.

    Returns:
        int: Number of true positives,
        int: Number of false positives,
        int: Number of false negatives.
    """
    matches = iou > threshold
    true_positives = np.sum(matches, axis=0) >= 1  # Correct objects
    # true_positives = np.sum(matches, axis=1) == 1  # Correct objects
    false_positives = np.sum(matches, axis=0) == 0  # Missed objects
    false_negatives = np.sum(matches, axis=1) == 0  # Extra objects
    tp, fp, fn = (
        np.sum(true_positives),
        np.sum(false_positives),
        np.sum(false_negatives),
    )
    return tp, fp, fn


def iou_map(truths, preds, verbose=0, ious=None):
    """
    Computes the metric for the competition.
    Masks contain the segmented pixels where each object has one value associated,
    and 0 is the background.

    Args:
        truths (list of masks): Ground truths.
        preds (list of masks): Predictions.
        verbose (int, optional): Whether to print infos. Defaults to 0.

    Returns:
        float: mAP.
    """
    if ious is None:
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


def evaluate_results(
    dataset,
    results,
    thresholds_conf=0.5,
    thresholds_mask=0.5,
    remove_overlap=False,
    num_classes=3,
    verbose=0,
):
    precs = [[] for _ in range(num_classes)]
    for idx in range(len(dataset)):
        masks, _, cell_type = post_process_preds(
            results[idx],
            thresholds_conf=thresholds_conf,
            thresholds_mask=thresholds_mask,
            remove_overlap=remove_overlap,
            num_classes=num_classes,
        )

        if not len(masks):
            precs[cell_type].append(0)
            continue

        rle_pred = [pycocotools.mask.encode(np.asarray(p, order='F')) for p in masks]
        rle_truth = dataset.encodings[idx].tolist()

        iou = pycocotools.mask.iou(rle_pred, rle_truth, [0] * len(rle_truth))
        score = iou_map(None, None, verbose=verbose, ious=[iou])
        precs[cell_type].append(score)

    return np.mean(np.concatenate(precs)), [np.mean(p) for p in precs if len(p)]
