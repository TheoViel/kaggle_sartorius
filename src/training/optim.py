import torch
import torch.nn as nn

from training.losses import SmoothCrossEntropyLoss, FocalTverskyLoss, lovasz_loss, DiceLoss


def define_optimizer(name, params, lr=1e-3):
    """
    Defines the loss function associated to the name.
    Supports optimizers from torch.nn.

    Args:
        name (str): Optimizer name.
        params (torch parameters): Model parameters
        lr (float, optional): Learning rate. Defaults to 1e-3.

    Raises:
        NotImplementedError: Specified optimizer name is not supported.

    Returns:
        torch optimizer: Optimizer
    """
    try:
        optimizer = getattr(torch.optim, name)(params, lr=lr)
    except AttributeError:
        raise NotImplementedError

    return optimizer


class SartoriusLoss(nn.Module):
    """
    Loss for the problem
    """
    def __init__(self, config):
        """
        Constructor
        Args:
            config (dict): Loss config.
        """
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss(reduction="none")
        self.ce = SmoothCrossEntropyLoss()
        self.focal_tversky = FocalTverskyLoss()
        self.lovasz = lovasz_loss
        self.dice = DiceLoss()

        self.w_seg_loss = config["w_seg_loss"]

        self.w_bce = config["w_bce"]
        self.w_dice = config["w_dice"]
        self.w_focal = config["w_focal"]
        self.w_lovasz = config["w_lovasz"]

    def compute_seg_loss(self, pred, truth):
        """
        Computes the auxiliary segmentation loss.
        Args:
            preds (list of torch tensors [BS x h_i x w_i]): Predicted masks.
            truth (torch tensor [BS x H x W]): Ground truth mask.
        Returns:
            torch tensor [BS]: Loss value.
        """
        loss = 0

        if self.w_bce:
            loss += self.w_bce * self.bce(pred, truth).mean(-1).mean(-1).mean(-1)

        if self.w_focal:
            loss += self.w_focal * self.focal_tversky(pred, truth).mean(-1)

        if self.w_lovasz:
            loss += self.w_lovasz * self.lovasz(pred, truth)

        if self.w_dice:
            loss += self.w_dice * self.dice(pred, truth)

        return loss

    def compute_cls_loss(self, pred, truth):
        """
        Computes the study loss. Handles mixup / cutmix.
        Args:
            preds (list of torch tensors or torch tensor [BS x num_classes]): Predictions.
            truth (torch tensor [BS x num_classes]): Ground truth.
        Returns:
            torch tensor [BS]: Loss value.
        """
        return self.bce(pred.float().view(truth.size()), truth.float())

    def __call__(
        self, pred_mask, pred_cls, y_mask, y_cls
    ):
        """
        Computes the overall loss.
        Args:

        Returns:
            torch tensor [BS]: Loss value.
        """
        seg_loss = self.compute_seg_loss(pred_mask, y_mask)

        if self.w_seg_loss < 1:
            cls_loss = self.compute_cls_loss(pred_cls, y_cls)
        else:
            cls_loss = 0

        return self.w_seg_loss * seg_loss + (1 - self.w_seg_loss) * cls_loss
