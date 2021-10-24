import numpy as np

from scipy import ndimage as ndi
from skimage.segmentation import watershed, relabel_sequential
from skimage.feature import peak_local_max
from skimage import measure


def remove_padding(pred, shape):
    """
    TODO
    pred in format ... x H x W

    Args:
        pred ([type]): [description]
        shape ([type]): [description]

    Returns:
        [type]: [description]
    """
    if pred.shape[-1] != shape[-1]:
        padding = pred.shape[-1] - shape[-1]
        pred = pred[..., padding // 2: - padding // 2]
    if pred.shape[-2] != shape[-2]:
        padding = pred.shape[-2] - shape[-2]
        pred = pred[..., padding // 2: - padding // 2, :]

    return pred


def remove_small_components(pred_i, min_size=10, verbose=0):
    for i in range(1, pred_i.max() + 1):
        if (pred_i == i).sum() < min_size:
            pred_i[pred_i == i] = 0
            if verbose:
                print(f'Removed component {i}')

    pred_i, _, _ = relabel_sequential(pred_i)

    return pred_i


def post_process_shsy5y(pred):
    distance = pred[2] * (1 - pred[1]) * pred[0]

    coords = peak_local_max(distance, min_distance=5, labels=pred[0] > 0.5, exclude_border=False)

    mask = np.zeros(distance.shape, dtype=bool)
    mask[tuple(coords.T)] = True
    markers, _ = ndi.label(mask)

    return watershed(-distance, markers, mask=(pred[0] > 0.5).astype(int))


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
            pred_i = remove_small_components(pred_i, min_size=100)
        elif cell_type == "cort":
            pred_i = post_process_cort(pred)
            pred_i = remove_small_components(pred_i, min_size=50)
        else:
            pred_i = post_process_astro(pred)
            pred_i = remove_small_components(pred_i, min_size=300)

        preds_instance.append(pred_i)
    return preds_instance
