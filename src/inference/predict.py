import torch

from data.loader import define_loaders


def predict(dataset, model, batch_size=16, device="cuda", mode="test"):
    """
    Performs inference on a dataset.
    The "mode" argument handles a small change in output format during inference.

    Args:
        dataset (SartoriusDataset): Inference dataset.
        model (torch Model): Detection model.
        batch_size (int, optional): Batch size. Defaults to 16.
        device (str, optional): Training device. Defaults to "cuda".
        mode (str, optional): Inference mode, can be "test" or "val". Defaults to "test".

    Returns:
        list of tuples: Results in the MMDet format [(boxes, masks), ...].
    """
    loader = define_loaders(None, dataset, val_bs=batch_size, num_workers=0)[1]
    all_results = []

    model.eval()
    with torch.no_grad():
        for i, batch in enumerate(loader):
            # print(dataset.cell_types[i], end=" ")
            if mode == "test":
                boxes, masks = model(**batch, return_loss=False, rescale=True)
                all_results.append((boxes.cpu().numpy(), masks.cpu().numpy()))
            else:
                results = model(**batch, return_loss=False, rescale=True)
                all_results += results

    return all_results
