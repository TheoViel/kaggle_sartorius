import torch
import numpy as np
from tqdm.notebook import tqdm  # noqa

from data.loader import define_loaders

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
    _, loader = define_loaders(
        None, dataset, val_bs=batch_size
    )
    all_results = []

    model.eval()
    with torch.no_grad():
        for batch in loader:

            n = len(batch['img_metas'].data[0])
            for i in range(n):
                batch['img_metas'].data[0][i]['scale_factor'] = np.ones(n)

            if not use_tta:
                for b in batch:
                    batch[b] = [batch[b]]  # no tta
            else:
                raise NotImplementedError

            results = model(**batch, return_loss=False)

            all_results += results

    return all_results
