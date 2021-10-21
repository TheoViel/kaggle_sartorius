import torch
from torch.utils.data import DataLoader

FLIPS = [[-1], [-2], [-2, -1]]


def get_flip_tta(x, use_rot=False):
    """
    Augments data with flipping.

    Args:
        x (torch tensor [BS x C x H x W]): Batch.
        use_rot (bool, optional): Whether to use rotations as well. Defaults to False.

    Returns:
        list of N torch tensors [N x BS x C x H x W]: Augmented images. N = 4 if use_rot, else 8.
    """
    x_hflip = x.flip([-1])
    x_vflip = x.flip([-2])
    x_hvflip = x.flip([-1, -2])

    if not use_rot:
        return [x, x_hflip, x_vflip, x_hvflip]

    x_rot = x.rot90(dims=(3, 2))
    x_rot_hflip = x_rot.flip([-1])
    x_rot_vflip = x_rot.flip([-2])
    x_rot_hvflip = x_rot.flip([-1, -2])

    return [x, x_hflip, x_vflip, x_hvflip, x_rot, x_rot_hflip, x_rot_vflip, x_rot_hvflip]


def predict(dataset, model, tta_fct=None, activation="sigmoid", device="cuda"):
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
    loader = DataLoader(dataset, batch_size=1, shuffle=False, pin_memory=True)
    preds = []

    model.eval()
    with torch.no_grad():
        for x, _ in loader:
            x = x.to(device)

            if tta_fct is not None:
                y_pred = []
                x_tta = tta_fct(x)
                x_tta = torch.cat(x_tta, 0)
                y_pred = model(x_tta).detach().mean(0)
            else:
                y_pred = model(x).detach()[0]

            if activation == "sigmoid":
                y_pred = torch.sigmoid(y_pred)

            preds.append(y_pred.cpu().numpy())

    return preds
