import cv2
import numpy as np
from scipy.ndimage import distance_transform_edt
from skimage.morphology import binary_erosion, dilation, disk  # noqa


def mask_to_contour(mask, width=1, pad=1):
    """
    Converts masks contours.

    Args:
        masks (numpy array [H x W x C]): Masks.
        axes (list, optional): Which channels to compute the contours for. Defaults to None.

    Returns:
        numpy array [H x W x C]: Contours.
    """
    mask = np.pad(mask, pad, mode="reflect")
    img = np.zeros(mask.shape)

    for i in range(1, int(np.max(mask)) + 1):
        m = ((mask == i) * 255).astype(np.uint8)
        contours, _ = cv2.findContours(m, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        cv2.polylines(img, contours, True, (i, i, i), width)

    return img[pad:-pad, pad:-pad].astype(np.uint16)


def get_distances(contours, masks, scale_factor=10):
    """
    Computes the distance map to a contour. It uses scipy's function.
    For visualization, the distances are clipped to [0, 1] and normalized by scale_factor.

    Args:
        contours (numpy array [H x W x C]): Contours.
        masks (numpy array [H x W x C]): Masks.
        scale_factor (int, optional): To divide the distance before clipping. Defaults to 100.

    Returns:
        numpy array [H x W x C]: Distance maps.
    """
    distances = np.zeros((masks.shape))

    if np.sum(masks):
        distances = distance_transform_edt(contours == 0) * (masks > 0)
        distances = np.clip(distances / scale_factor, 0, 1)

    return (distances * 10000).astype(np.uint16)


def keep_cell_to_cell_only(contour, mask):
    mask_eroded = binary_erosion((mask > 0).astype(int), disk(2))
    border = contour * mask_eroded
#     border = dilation(border, disk(1))
    return border
