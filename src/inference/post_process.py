import numpy as np

from scipy import ndimage as ndi
from skimage.segmentation import watershed
from skimage.feature import peak_local_max
from skimage import measure


def remove_padding(pred, truth):
    assert len(pred.shape) <= 3, "Pred shape too big"
    if pred.shape[0] == truth.shape[0]:  # padding on first two axes
        if pred.shape[0] != truth.shape[0]:
            padding = pred.shape[0] - truth.shape[0]
            pred = pred[padding // 2: - padding // 2]
        if pred.shape[1] != truth.shape[1]:
            padding = pred.shape[1] - truth.shape[1]
            pred = pred[:, padding // 2: - padding // 2]
    elif pred.shape[-1] == truth.shape[-1]:  # padding on last two axes
        if pred.shape[-1] != truth.shape[-1]:
            padding = pred.shape[-1] - truth.shape[-1]
            pred = pred[..., padding // 2: - padding // 2]
        if pred.shape[-2] != truth.shape[-2]:
            padding = pred.shape[-2] - truth.shape[-2]
            pred = pred[..., padding // 2: - padding // 2, :]
    else:
        raise NotImplementedError

    return pred


def post_process_shsy5y(pred):
    distance = pred[2] * (1 - pred[1]) * pred[0]

    coords = peak_local_max(distance, min_distance=5, labels=pred[0] > 0.3, exclude_border=False)

    mask = np.zeros(distance.shape, dtype=bool)
    mask[tuple(coords.T)] = True
    markers, _ = ndi.label(mask)

    return watershed(-distance, markers, mask=(pred[0] > 0.3).astype(int))


def post_process_cort(pred):
    distance = pred[0] * (1 - pred[1])  # * pred[2]
    image = (distance > 0.5).astype(int)
    y_pred = measure.label(image, neighbors=8, background=0)
    props = measure.regionprops(y_pred)
    for j in range(len(props)):
        if props[j].area < 12:
            y_pred[y_pred == j + 1] = 0
    y_pred = measure.label(y_pred, neighbors=8, background=0)

    mask = (pred[0] > 0.5).astype(int)
    return watershed(pred[0], y_pred, mask=mask, watershed_line=True)


def post_process_astro(pred):
    distance = pred[2] * (1 - pred[1]) * pred[0]

    coords = peak_local_max(distance, min_distance=20, labels=pred[0] > 0.5, exclude_border=False)

    mask = np.zeros(distance.shape, dtype=bool)
    mask[tuple(coords.T)] = True
    markers, _ = ndi.label(mask)

    return watershed(-distance, markers, mask=(pred[0] > 0.5).astype(int))


def preds_to_instance(preds, cell_types):
    preds_instance = []
    for pred, cell_type in zip(preds, cell_types):
        if cell_type == "shsy5y":
            pred_i = post_process_shsy5y(pred)
        elif cell_type == "cort":
            pred_i = post_process_cort(pred)
        else:
            pred_i = post_process_astro(pred)

        preds_instance.append(pred_i)
    return preds_instance
