import torch

from data.loader import define_loaders


def predict(dataset, model, batch_size=16, device="cuda", mode="test"):
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
    loader = define_loaders(None, dataset, val_bs=batch_size, num_workers=0)[1]
    all_results = []

    model.eval()
    with torch.no_grad():
        for batch in loader:
            if mode == "test":
                boxes, masks = model(**batch, return_loss=False, rescale=True)
                all_results.append((boxes.cpu().numpy(), masks.cpu().numpy()))
            else:
                results = model(**batch, return_loss=False, rescale=True)
                all_results += results

    return all_results
