import torch
import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR
from transformers import get_linear_schedule_with_warmup

from training.losses import SmoothCrossEntropyLoss, FocalTverskyLoss, lovasz_loss, DiceLoss

NO_DECAY = [
    "bias", "LayerNorm.weight", "absolute_pos_embed", "relative_position_bias_table", "norm", "bn"
]


def define_optimizer(name, model, lr=1e-3, weight_decay=0):
    """
    Defines the loss function associated to the name.
    Supports optimizers from torch.nn.
    TODO

    Args:
        name (str): Optimizer name.
        params (torch parameters): Model parameters
        lr (float, optional): Learning rate. Defaults to 1e-3.

    Raises:
        NotImplementedError: Specified optimizer name is not supported.

    Returns:
        torch optimizer: Optimizer
    """
    opt_params = []
    for n, p in model.named_parameters():
        wd = 0 if any(nd in n for nd in NO_DECAY) else weight_decay
        opt_params.append(
            {"params": [p], "weight_decay": wd, "lr": lr}
        )

    try:
        optimizer = getattr(torch.optim, name)(opt_params, lr=lr)
    except AttributeError:
        raise NotImplementedError(f'Optimizer {name} not supported')

    return optimizer


def get_plateau_schedule_with_warmup(
    optimizer, num_warmup_steps, num_training_steps, decay_steps=[0.5], decay=0.1, last_epoch=-1
):
    """
    Linear warmup + divide lr by 10 at each decay_steps
    """
    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:  # warmup
            return float(current_step) / float(max(1, num_warmup_steps))

        for i, decay_step in enumerate(decay_steps[::-1] + [0]):
            if current_step > decay_step * num_training_steps:
                break
        return decay ** (len(decay_steps) - i)

    return LambdaLR(optimizer, lr_lambda, last_epoch)


def define_scheduler(scheduler, optimizer, num_warmup_steps, num_training_steps):
    if scheduler == "linear":
        return get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps, num_training_steps
        )
    elif scheduler == "plateau":
        return get_plateau_schedule_with_warmup(
            optimizer, num_warmup_steps, num_training_steps, decay_steps=[0.8, 0.95]
        )
    else:
        raise NotImplementedError


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
        truth[:, :2] = (truth[:, :2] > 0).float()  # ignore instance id

        loss = 0
        for i in range(3):
            if self.w_bce[i]:
                loss += self.w_bce[i] * self.bce(pred[:, i], truth[:, i]).mean((1, 2))

            if self.w_focal[i]:
                loss += self.w_focal[i] * self.focal_tversky(pred[:, i], truth[:, i]).mean(-1)

            if self.w_lovasz[i]:
                loss += self.w_lovasz[i] * self.lovasz(pred[:, i], truth[:, i])

            if self.w_dice[i]:
                loss += self.w_dice[i] * self.dice(pred[:, i], truth[:, i])

        return torch.div(loss, 3)

    def compute_cls_loss(self, pred, truth):
        """
        Computes the study loss. Handles mixup / cutmix.
        Args:
            preds (list of torch tensors or torch tensor [BS x num_classes]): Predictions.
            truth (torch tensor [BS x num_classes]): Ground truth.
        Returns:
            torch tensor [BS]: Loss value.
        """
        return self.ce(pred, truth.long())

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
