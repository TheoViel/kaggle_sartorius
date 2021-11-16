import cv2
import skimage
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
import plotly.graph_objects as go

from scipy import ndimage
from mmdet.core import BitmapMasks
from matplotlib.patches import Rectangle

from utils.metrics import compute_iou


GREEN = (56 / 255, 200 / 255, 100 / 255)
BLUE = (32 / 255, 50 / 255, 155 / 255)
RED = (238 / 255, 97 / 255, 55 / 255)


def get_random_color():
    color = tuple(np.random.random(size=3))
    while np.max(color) - np.min(color) < 0.2:
        color = tuple(np.random.random(size=3))
    return color


def plot_sample(img, mask=None, boxes=[], width=1, plotly=False):
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

    colors = []

    if isinstance(mask, BitmapMasks):
        mask = mask.masks.astype(int)
        for i in range(len(mask)):
            mask[i] *= (i + 1)

    if mask is not None:
        if len(mask.shape) == 3:
            if mask.max() == 1:
                for i in range(len(mask)):
                    mask[i] *= (i + 1)
            mask = mask.max(0)

        for i in range(1, int(np.max(mask)) + 1):
            m = ((mask == i) * 255).astype(np.uint8)
            color = get_random_color()
            colors.append(color)

            contours, _ = cv2.findContours(m, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
            cv2.polylines(img_, contours, True, color, width)

    if not plotly:
        plt.imshow(img_)

        # Add boxes
        for i, box in enumerate(boxes):
            color = colors[i] if len(colors) else get_random_color()
            rect = Rectangle(
                (box[0], box[1]), box[2] - box[0], box[3] - box[1],
                linewidth=1, edgecolor=color, facecolor='none', alpha=0.5
            )
            plt.gca().add_patch(rect)

    if plotly:
        return px.imshow(img_)


def get_centers(mask):
    centers = []
    for i in range(1, int(np.max(mask)) + 1):
        centers.append(ndimage.measurements.center_of_mass(mask == i))
    return np.array(centers)


def plot_preds_iou(img, preds, truths, boxes=None, boxes_2=None, width=1, plot_tp=True):
    """
    Plots the contours of a given mask.
    TODO
    """
    if img.max() > 1:
        img = (img / 255).astype(float)

    if len(img.shape) == 2:
        img = np.stack([img, img, img], -1)

    preds, _, _ = skimage.segmentation.relabel_sequential(preds)
    truths, _, _ = skimage.segmentation.relabel_sequential(truths)
    ious = compute_iou(truths, preds)

    centers_pred = get_centers(preds)

    img_ = img.copy()

    # Add boxes
    if boxes is not None:
        for box in boxes:
            color = (0.5, 0, 0, 0.5)
            img_ = cv2.rectangle(img_, (box[0], box[1]), (box[2], box[3]), color=color, thickness=1)

    # Add boxes
    if boxes_2 is not None:
        for box in boxes_2:
            color = (0., 0, 0.5, 0.5)
            img_ = cv2.rectangle(img_, (box[0], box[1]), (box[2], box[3]), color=color, thickness=1)

    # Plot preds
    for i in range(1, int(np.max(preds)) + 1):
        m = ((preds == i) * 255).astype(np.uint8)

        if not plot_tp and ious.max(0)[i - 1] > 0.5:
            continue

        contours, _ = cv2.findContours(m, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        cv2.polylines(img_, contours, True, BLUE, width)

    # Plot truths
    for i in range(1, int(np.max(truths)) + 1):
        m = ((truths == i) * 255).astype(np.uint8)
        color = GREEN if ious.max(1)[i - 1] > 0.5 else RED

        contours, _ = cv2.findContours(m, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        cv2.polylines(img_, contours, True, color, width)

    fig = px.imshow(img_)

    # Plot IoUs
    fig.add_trace(
        go.Scatter(
            x=centers_pred[:, 1],
            y=centers_pred[:, 0],
            mode='markers',
            name="Pred Center",
            text=[f"IoU : {m:.2f}" for m in ious.max(0)],
            marker_color='rgba(0, 0, 255, .5)'
        )
    )

    fig.update_xaxes(range=[0, img.shape[1]])
    fig.update_yaxes(range=[img.shape[0], 0])

    return fig
