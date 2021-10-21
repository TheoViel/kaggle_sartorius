import cv2
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt


def plot_sample(img, mask=None, width=1, plotly=False):
    """
    Plots the contours of a given mask.

    Args:
        img (numpy array [H x W]): Image.
        mask (numpy array [H x W x C]): Masks.
        width (int, optional): Contour width. Defaults to 1.

    Returns:
        img (numpy array [H x W]): Image with contours.
    """

    if img.max() > 1:
        img = (img / 255).astype(float)

    if len(img.shape) == 2:
        img = np.stack([img, img, img], -1)

    img_ = img.copy()
    if mask is not None:
        for i in range(1, int(np.max(mask)) + 1):
            m = ((mask == i) * 255).astype(np.uint8)
            color = tuple(np.random.random(size=3))

            contours, _ = cv2.findContours(m, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
            cv2.polylines(img_, contours, True, color, width)

    if plotly:
        return px.imshow(img_)
    else:
        plt.imshow(img_)
