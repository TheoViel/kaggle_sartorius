import numpy as np

# from scipy import ndimage as ndis
from skimage.segmentation import relabel_sequential
from cellpose.dynamics import compute_masks


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


def post_process_shsy5y(y):
    masks, _, _ = compute_masks(
        y[0],
        y[1:],
        bd=np.zeros_like(y[0]),
        niter=1,
        mask_threshold=0.5,
        # diam_threshold=100,
        # flow_threshold=-1,
        min_size=10,
        omni=False,
    )

    return masks


def post_process_cort(y):
    masks, _, _ = compute_masks(
        y[0],
        y[1:],
        bd=np.zeros_like(y[0]),
        niter=1,
        mask_threshold=0.5,
        # diam_threshold=100,
        # flow_threshold=-1,
        min_size=10,
        omni=False,
    )

    return masks


def post_process_astro(y):
    masks, _, _ = compute_masks(
        y[0],
        y[1:],
        bd=np.zeros_like(y[0]),
        niter=1,
        mask_threshold=0.5,
        # diam_threshold=100,
        # flow_threshold=-1,
        min_size=10,
        omni=False,
    )

    return masks


def preds_to_instance(preds, cell_types):
    preds_instance = []
    for pred, cell_type in zip(preds, cell_types):
        pred(pred.shape)
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
