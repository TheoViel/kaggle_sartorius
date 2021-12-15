import torch
import numpy as np
from tqdm.notebook import tqdm

from data.loader import define_loaders
from data.transforms import define_pipelines
from data.dataset import SartoriusInferenceDataset

from model_zoo.models import define_model
from model_zoo.ensemble import EnsembleModel

from utils.rle import rle_encoding
from utils.torch import load_model_weights

from mmcv.parallel import MMDataParallel
from inference.post_process import remove_overlap_naive, corrupt_mask, mask_nms


def process_masks(
    boxes,
    masks,
    thresholds_mask,
    thresholds_nms,
    thresholds_conf,
    remove_overlap=True,
    corrupt=False
):
    """
    Complete processing function for a single (masks, boxes) pair.

    Args:
        masks (np array [n x H x W]): Masks.
        boxes (np array [n x 5]): Boxes.
        thresholds_mask (list of float [3]): Thresholds per class for masks.
        thresholds_nms (list of float [3]): Thresholds per class for nms.
        thresholds_conf (list of float [3]): Thresholds per class for confidence.
        remove_overlap (bool, optional): Whether to remove overlap. Defaults to True.
        corrupt (bool, optional): Whether to corrupt astro masks. Defaults to True.

    Returns:
        np arrays [m x H x W]: Masks.
        np arrays [m x 5]: Boxes.
        int : Cell type.
    """

    # Cell type
    cell = np.argmax(np.bincount(boxes[:, 5].astype(int)))

    # Thresholds
    thresh_mask = (
        thresholds_mask if isinstance(thresholds_mask, (float, int))
        else thresholds_mask[cell]
    )
    thresh_nms = (
        thresholds_nms if isinstance(thresholds_nms, (float, int))
        else thresholds_nms[cell]
    )
    thresh_conf = (
        thresholds_conf if isinstance(thresholds_conf, (float, int))
        else thresholds_conf[cell]
    )

    # Binarize
    masks = masks > (thresh_mask * 255)

    # Sort by decreasing conf
    order = np.argsort(boxes[:, 4])[::-1]
    masks = masks[order]
    boxes = boxes[order]

    # Remove low confidence
    last = (
        np.argmax(boxes[:, 4] < thresh_conf) if np.min(boxes[:, 4]) < thresh_conf
        else len(boxes)
    )
    masks = masks[:last]
    boxes = boxes[:last]

    # NMS
    if thresh_nms > 0:
        masks, boxes, _ = mask_nms(masks, boxes, thresh_nms)

    # Corrupt
    if corrupt and cell == 1:  # astro
        masks = np.array([corrupt_mask(mask)[0] for mask in masks])

    # Remove overlap
    if remove_overlap:
        masks = remove_overlap_naive(masks)

    return masks, boxes, cell


def predict_and_process(
    dataset,
    model,
    thresholds_mask,
    thresholds_nms,
    thresholds_conf,
    corrupt=True,
    remove_overlap=True,
    device="cuda"
):
    """
    Inference + post-processing + convert to rle.

    Args:
        dataset (InferenceDataset): Inference dataset.
        model (torch model): Segmentation model.
        thresholds_mask (list of float [3]): Thresholds per class for masks.
        thresholds_nms (list of float [3]): Thresholds per class for nms.
        thresholds_conf (list of float [3]): Thresholds per class for confidence.
        corrupt (bool, optional): Whether to corrupt astro masks. Defaults to True.
        remove_overlap (bool, optional): Whether to remove overlap. Defaults to True.
        device (str, optional): Device. Defaults to "cuda".

    Returns:
        list of strings: Predicted RLEs.
    """
    loader = define_loaders(None, dataset, val_bs=1, num_workers=0)[1]
    rles = []

    model.eval()
    with torch.no_grad():
        for batch in tqdm(loader):
            boxes, masks = model(**batch, return_loss=False, rescale=True)
            boxes = boxes.cpu().numpy()
            masks = masks.cpu().numpy()

            masks, boxes, cell_type = process_masks(
                boxes,
                masks,
                thresholds_mask,
                thresholds_nms,
                thresholds_conf,
                remove_overlap=remove_overlap,
                corrupt=corrupt
            )

            rles.append([rle_encoding(mask) for mask in masks])

    return rles


def inference(
    df,
    configs,
    weights,
    ensemble_config,
    thresholds_mask,
    thresholds_nms,
    thresholds_conf,
    corrupt=True,
    remove_overlap=True
):
    """
    Inference function for the test data.

    Args:
        df (pd Dataframe): Metadata.
        configs (list of Configs): Configs.
        weights (list of strings): Weights.
        ensemble_config (dict): Config of the ensemble model.
        thresholds_mask (list of float [3]): Thresholds per class for masks.
        thresholds_nms (list of float [3]): Thresholds per class for nms.
        thresholds_conf (list of float [3]): Thresholds per class for confidence.
        corrupt (bool, optional): Whether to corrupt astro masks. Defaults to True.
        remove_overlap (bool, optional): Whether to remove overlap. Defaults to True.

    Returns:
        list of strings: Predicted RLEs.
    """

    pipelines = define_pipelines(configs[0].data_config)

    models, names = [], []
    for config, fold_weights in zip(configs, weights):
        for weight in fold_weights:
            model = define_model(
                config.model_config, encoder=config.encoder, verbose=0
            )
            model = load_model_weights(model, weight, verbose=0)
            models.append(model)
            names.append(weight.split('/')[-1])

    dataset = SartoriusInferenceDataset(
        df,
        transforms=pipelines['test_tta'] if ensemble_config["use_tta"] else pipelines['test']
    )

    model = MMDataParallel(
        EnsembleModel(
            models,
            ensemble_config,
            names=names,
        )
    )

    rles = predict_and_process(
        dataset,
        model,
        thresholds_mask,
        thresholds_nms,
        thresholds_conf,
        device=config.device,
        corrupt=corrupt,
        remove_overlap=remove_overlap,
    )

    return rles
