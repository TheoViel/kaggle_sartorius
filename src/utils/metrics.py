import skimage
import numpy as np


def dice_scores_img(pred, truth, eps=1e-8):
    """
    Dice metric for a single image as array.

    Args:
        pred (np array): Predictions.
        truth (np array): Ground truths.
        eps (float, optional): epsilon to avoid dividing by 0. Defaults to 1e-8.

    Returns:
        np array : dice value for each class.
    """
    pred = pred.reshape(-1) > 0
    truth = truth.reshape(-1) > 0
    intersect = (pred & truth).sum(-1)
    union = pred.sum(-1) + truth.sum(-1)

    dice = (2.0 * intersect + eps) / (union + eps)
    return dice


def dice_scores_img_tensor(pred, truth, eps=1e-8):
    """
    Dice metric for a single image as tensor.

    Args:
        pred (torch tensor): Predictions.
        truth (torch tensor): Ground truths.
        eps (float, optional): epsilon to avoid dividing by 0. Defaults to 1e-8.

    Returns:
        np array : dice value for each class.
    """
    pred = pred.view(-1) > 0
    truth = truth.contiguous().view(-1) > 0
    intersect = (pred & truth).sum(-1)
    union = pred.sum(-1) + truth.sum(-1)

    dice = (2.0 * intersect + eps) / (union + eps)
    return float(dice)


def dice_score(pred, truth, eps=1e-8, threshold=0.5):
    """
    Dice metric. Only classes that are present are weighted.

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


def dice_score_tensor(pred, truth, eps=1e-8, threshold=0.5):
    """
    Dice metric for tensors. Only classes that are present are weighted.

    Args:
        pred (torch tensor): Predictions.
        truth (torch tensor): Ground truths.
        eps (float, optional): epsilon to avoid dividing by 0. Defaults to 1e-8.
        threshold (float, optional): Threshold for predictions. Defaults to 0.5.

    Returns:
        float: dice value.
    """
    pred = (pred.view((truth.size(0), -1)) > threshold).int()
    truth = truth.view((truth.size(0), -1)).int()
    intersect = (pred + truth == 2).sum(-1)
    union = pred.sum(-1) + truth.sum(-1)
    dice = (2.0 * intersect + eps) / (union + eps)
    return dice.mean()


def tweak_threshold(mask, pred):
    """
    Tweaks the threshold to maximise the score.

    Args:
        mask (torch tensor): Ground truths.
        pred (torch tensor): Predictions.

    Returns:
        float: Best threshold.
        float: Best score.
    """
    thresholds = []
    scores = []
    for threshold in np.linspace(0.2, 0.7, 11):

        dice_score = dice_scores_img_tensor(pred=pred > threshold, truth=mask)
        thresholds.append(threshold)
        scores.append(dice_score)

    return thresholds[np.argmax(scores)], np.max(scores)


def compute_iou(labels, y_pred):
    """
    Computes the IoU for instance labels and predictions.

    Args:
        labels (np array): Labels.
        y_pred (np array): predictions

    Returns:
        np array: IoU matrix, of size true_objects x pred_objects.
    """
    labels, _, _ = skimage.segmentation.relabel_sequential(labels)
    y_pred, _, _ = skimage.segmentation.relabel_sequential(y_pred)

    true_objects = len(np.unique(labels))
    pred_objects = len(np.unique(y_pred))

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

    return intersection / (union + 1e-6)


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


def iou_map(truths, preds, verbose=0):
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

        p = tps / (tps + fps + fns)
        prec.append(p)

        if verbose:
            print("{:1.3f}\t{}\t{}\t{}\t{:1.3f}".format(t, tps, fps, fns, p))

    if verbose:
        print("AP\t-\t-\t-\t{:1.3f}".format(np.mean(prec)))

    return np.mean(prec)
