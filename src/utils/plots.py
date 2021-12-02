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
    """
    Returns a random color thats visible on a gray background.

    Returns:
        tuple of floats(3): Color rgb code between 0 and 1.
    """
    color = tuple(np.random.random(size=3))
    while np.max(color) - np.min(color) < 0.2:
        color = tuple(np.random.random(size=3))
    return color


def plot_sample(img, mask=None, boxes=[], width=1, plotly=False):
    """
    Plots a sample

    Args:
        img (numpy array [H x W]): Image.
        mask (numpy array [n x H x W or H x W], optional): Masks. Defaults to None
        boxes (numpy array [n x 4], optional): Boxes. Defaults to [].
        width (int, optional): Contour width. Defaults to 1.
        plotly (bool, optional): Whether to use plotly instead of matplotlib. Defaults to False.

    Returns:
        plotly.express figure : Plotly figure to display if plotly is used, else None.
    """
    if img.max() > 1:
        img = (img / 255).astype(float)

    if len(img.shape) == 2:
        img = np.stack([img, img, img], -1)

    img_ = img.copy()

    colors = []

    if isinstance(mask, BitmapMasks):
        mask = mask.masks.astype(int)

    if np.max(mask) == 1:
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
    """
    Gets the centers of a mask cells.

    Args:
        mask (np array [H x W]): Mask.

    Returns:
        np array [n x 2]: Centers.
    """
    centers = []
    for i in range(1, int(np.max(mask)) + 1):
        centers.append(ndimage.measurements.center_of_mass(mask == i))
    return np.array(centers)


def plot_preds_iou(
    img, preds, truths, boxes=None, boxes_2=None, iou_thresh=0.5, width=1, plot_tp=True
):
    """
    Plots the prediction using a color code depending on the IoU:
    - Green = TP : Ground truth cell with an IoU > iou_thresh with a prediction
    - Red = FN : IoU  truth cell with no IoU > iou_thresh with any prediction
    - Blue = Predictions

    Args:
        img (numpy array [H x W]): Image.
        preds (numpy array [n1 x H x W or H x W]): Predicted Masks.
        truths (numpy array [n2 x H x W or H x W]): GT Masks.
        boxes (numpy array [m1 x 4], optional): Boxes to display in red. Defaults to None.
        boxes_2 (numpy array [m2 x 4], optional): Boxes to display in blue. Defaults to None.
        iou_thresh (float, optional): Threshold to determine a hit or a miss. Defaults to 0.5
        width (int, optional): Contour width. Defaults to 1.
        plot_tp (bool, optional): Whether to display TP predictions. Defaults to True.

    Returns:
        plotly.express figure : Plotly figure to display.
    """
    if img.max() > 1:
        img = (img / 255).astype(float)

    if len(img.shape) == 2:
        img = np.stack([img, img, img], -1)

    if len(preds.shape) == 3:
        if preds.max() == 1:
            for i in range(len(preds)):
                preds[i] *= (i + 1)
        preds = preds.max(0)

    if len(truths.shape) == 3:
        if truths.max() == 1:
            for i in range(len(truths)):
                truths[i] *= (i + 1)
        truths = truths.max(0)

    preds, _, _ = skimage.segmentation.relabel_sequential(preds)
    truths, _, _ = skimage.segmentation.relabel_sequential(truths)
    ious = compute_iou(truths, preds)

    centers_pred = get_centers(preds)

    img_ = img.copy()

    # Plot preds
    for i in range(1, int(np.max(preds)) + 1):
        m = ((preds == i) * 255).astype(np.uint8)

        if not plot_tp and ious.max(0)[i - 1] > iou_thresh:
            continue

        contours, _ = cv2.findContours(m, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        cv2.polylines(img_, contours, True, BLUE, width)

    # Plot truths
    for i in range(1, int(np.max(truths)) + 1):
        m = ((truths == i) * 255).astype(np.uint8)
        color = GREEN if ious.max(1)[i - 1] > iou_thresh else RED

        contours, _ = cv2.findContours(m, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        cv2.polylines(img_, contours, True, color, width)

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
