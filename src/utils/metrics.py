import skimage
import numpy as np
from tqdm.notebook import tqdm  # noqa
from multiprocessing import Pool


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
    true_positives = np.sum(matches, axis=1) == 1  # Correct objects
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

    prec = []
    for t in np.arange(0.5, 1.0, 0.05):
        tps, fps, fns = 0, 0, 0
        for iou in ious:
            tp, fp, fn = precision_at(t, iou)
            tps += tp
            fps += fp
            fns += fn

        p = tps / (tps + fps + fns) if tps else 0
        prec.append(p)

        if verbose:
            print("{:1.3f}\t{}\t{}\t{}\t{:1.3f}".format(t, tps, fps, fns, p))

    if verbose:
        print("AP\t-\t-\t-\t{:1.3f}".format(np.mean(prec)))

    return np.mean(prec)


def evaluate_results(dataset, results, threshold=0.5, num_classes=1, verbose=0):
    ious = []
    for idx in range(len(dataset)):

        # retrieve masks
        if num_classes == 1:
            # remove low confidence
            scores = np.concatenate(results[idx][0])[:, -1]
            last = np.argmax(scores < threshold) if len(scores) else 0
            if not last:
                continue

            masks_pred = np.array(results[idx][1][0][:last]).astype(int)
        else:
            raise NotImplementedError()
            masks_pred = np.concatenate(results[idx][1]).astype(int)

        masks_truth = dataset.masks[idx].masks

        # # convert to format for metric
        mask_truth = np.zeros(masks_truth.shape[1:], dtype=int)
        for i in range(len(masks_truth)):
            mask_truth = np.where(
                np.logical_and(masks_truth[i], mask_truth == 0), i + 1, mask_truth
            )

        mask_pred = np.zeros(masks_pred.shape[1:], dtype=int)
        for i in range(len(masks_pred)):
            mask_pred = np.where(
                np.logical_and(masks_pred[i], mask_pred == 0), i + 1, mask_pred
            )

        ious.append(compute_iou(mask_truth, mask_pred))

    return iou_map(None, None, verbose=verbose, ious=ious)


def compute_ious(idx, dataset, results, threshold):
    scores = np.concatenate(results[idx][0])[:, -1]
    last = np.argmax(scores < threshold) if len(scores) else 0
    if not last:
        return

    masks_pred = np.array(results[idx][1][0][:last]).astype(int)
    masks_truth = dataset.masks[idx].masks

    # convert to format for metric
    mask_truth = np.zeros(masks_truth.shape[1:], dtype=int)
    for i in range(len(masks_truth)):
        mask_truth = np.where(
            np.logical_and(masks_truth[i], mask_truth == 0), i + 1, mask_truth
        )

    mask_pred = np.zeros(masks_pred.shape[1:], dtype=int)
    for i in range(len(masks_pred)):
        mask_pred = np.where(
            np.logical_and(masks_pred[i], mask_pred == 0), i + 1, mask_pred
        )

    return compute_iou(mask_truth, mask_pred)


def compute_ious_(idx):
    return compute_ious(idx=idx, dataset=DATASET, results=RESULTS, threshold=THRESHOLD)


def evaluate_results_multiproc(dataset, results, threshold=0.5, verbose=0):
    global DATASET
    DATASET = dataset
    global RESULTS
    RESULTS = results
    global THRESHOLD
    THRESHOLD = threshold

    ious = []
    with Pool(processes=4) as p:
        for iou in p.map(compute_ious_, range(len(dataset))):
            if iou is not None:
                ious.append(iou)

    return iou_map(None, None, verbose=verbose, ious=ious)
