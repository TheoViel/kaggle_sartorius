import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm.notebook import tqdm  # noqa

from params import NUM_WORKERS

FLIPS = [[-1], [-2], [-2, -1]]


def predict(dataset, model, activations={}, batch_size=16, use_tta=False, device="cuda"):
    """
    Performs inference on an image.
    TODO

    Args:
        dataset (InferenceDataset): Inference dataset.
        model (torch model): Segmentation model.
        batch_size (int, optional): Batch size. Defaults to 32.
        tta (bool, optional): Whether to apply tta. Defaults to False.

    Returns:
        torch tensor [H x W]: Prediction on the image.
    """
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=NUM_WORKERS
    )
    all_preds_mask, all_preds_cls = [], []

    model.eval()
    with torch.no_grad():
        # for x, _, _ in tqdm(loader):
        for x, _, _ in loader:
            x = x.to(device)

            pred_mask, pred_cls = model(x)

            pred_mask, pred_cls = compute_activations(pred_mask, pred_cls, activations, detach=True)

            if use_tta:
                for f in FLIPS:
                    pred_mask_f, pred_cls_f = model(x.flip(f))
                    pred_mask_f, pred_cls_f = compute_activations(
                        pred_mask_f, pred_cls_f, activations, detach=True
                    )

                    pred_mask += pred_mask_f.flip(f)
                    pred_cls += pred_cls_f

                pred_mask = torch.div(pred_mask, len(FLIPS) + 1)
                pred_cls = torch.div(pred_cls, len(FLIPS) + 1)

            all_preds_mask.append(pred_mask.cpu().numpy())
            all_preds_cls.append(pred_cls.cpu().numpy())

    return np.concatenate(all_preds_mask), np.concatenate(all_preds_cls)


def compute_activations(pred_mask, pred_cls, activations, detach=False):
    """
    Applies activations TODO

    Args:
        pred_mask ([type]): [description]
        pred_cls ([type]): [description]
        activations ([type]): [description]
        detach (bool, optional): [description]. Defaults to False.

    Returns:
        [type]: [description]
    """
    if detach:
        pred_cls = pred_cls.detach()
        pred_mask = pred_mask.detach()

    if isinstance(activations, str):
        raise NotImplementedError  # TODO, handle string

    if 'mask' in activations.keys():
        if activations['mask'] == 'sigmoid':
            pred_mask = torch.sigmoid(pred_mask)

    if 'cls' in activations.keys():
        if activations['cls'] == 'softmax':
            pred_cls = torch.softmax(pred_cls, -1)
        elif activations['cls'] == 'sigmoid':
            pred_cls = torch.sigmoid(pred_cls)

    return pred_mask, pred_cls
