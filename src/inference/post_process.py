import numpy as np

from scipy import ndimage as ndi
from skimage.segmentation import watershed
from skimage.feature import peak_local_max
from skimage import measure


def remove_padding(pred, truth):
    if pred.shape[0] != truth.shape[0]:
        padding = pred.shape[0] - truth.shape[0]
        pred = pred[padding // 2: - padding // 2]
    if pred.shape[1] != truth.shape[1]:
        padding = pred.shape[1] - truth.shape[1]
        pred = pred[:, padding // 2: - padding // 2]

    return pred


def preds_to_instance(preds, threshold=0.5):
    preds_instance = []
    for pred in preds:
        if pred.shape[0] == 1:
            image = (pred[0] > threshold).astype(int)

            distance = ndi.distance_transform_edt(image)

            coords = peak_local_max(distance, min_distance=10, labels=image, exclude_border=False)
            mask = np.zeros(distance.shape, dtype=bool)
            mask[tuple(coords.T)] = True
            markers, _ = ndi.label(mask)

            pred_instance = watershed(-distance, markers, mask=(pred[0] > threshold).astype(int))

        else:
            image = ((pred[0] * (1 - pred[1])) > threshold).astype(int)
            y_pred = measure.label(image, neighbors=8, background=0)
            props = measure.regionprops(y_pred)
            for i in range(len(props)):
                if props[i].area < 12:
                    y_pred[y_pred == i + 1] = 0
            y_pred = measure.label(y_pred, neighbors=8, background=0)

            mask = (pred[0] > threshold).astype(int)
            pred_instance = watershed(pred[0], y_pred, mask=mask, watershed_line=True)

        preds_instance.append(pred_instance)

    return preds_instance
