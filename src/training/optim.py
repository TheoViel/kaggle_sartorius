import torch
from torch.optim.lr_scheduler import LambdaLR
from transformers import get_linear_schedule_with_warmup

NO_DECAY = [
    "bias", "LayerNorm.weight", "absolute_pos_embed", "relative_position_bias_table", "norm", "bn"
]


def define_optimizer(name, model, lr=1e-3, weight_decay=0):
    """
    Defines the optimizer.

    Args:
        name (str): Optimizer name.
        model (torch model): Model.
        lr (float, optional): Learning rate. Defaults to 1e-3.
        weight_decay (float, optional): Weight decay. Defaults to 0.

    Raises:
        NotImplementedError: Specified optimizer name is not supported.

    Returns:
        torch optimizer: Optimizer.
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
    optimizer, num_warmup_steps, num_training_steps, decay_props=[0.5], decay=0.1
):
    """
    Custom scheduler to reproduce the MMDet training configs :
    Linear warmup + multiply by decay at each decay_steps * num_training_steps.

    Args:
        optimizer (torch Optimizer): Optimizer.
        num_warmup_steps (int): Number of warmup steps.
        num_training_steps (int): Number of total training steps.
        decay_props (list, optional): Training proportions to decay the lr at. Defaults to [0.5].
        decay (float, optional): Coefficient to decay the learning rate by. Defaults to 0.1.
    """
    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:  # warmup
            return float(current_step) / float(max(1, num_warmup_steps))

        for i, decay_step in enumerate(decay_props[::-1] + [0]):
            if current_step > decay_step * num_training_steps:
                break
        return decay ** (len(decay_props) - i)

    return LambdaLR(optimizer, lr_lambda, last_epoch=-1)


def define_scheduler(scheduler, optimizer, num_warmup_steps, num_training_steps):
    """
    Defines a scheduler.
    Supports linear_schedule_with_warmup and the custom plateau_schedule_with_warmup.

    Args:
        scheduler (str): Scheduler name.
        optimizer (torch Optimizer): Optimizer.
        num_warmup_steps (int): Number of warmup steps.
        num_training_steps (int): Number of total training steps.

    Raises:
        NotImplementedError: Scheduler name not supported.

    Returns:
        LambdaLR: Scheduler.
    """
    if scheduler == "linear":
        return get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps, num_training_steps
        )
    elif scheduler == "plateau":
        return get_plateau_schedule_with_warmup(
            optimizer, num_warmup_steps, num_training_steps, decay_props=[0.8, 0.95]
        )
    else:
        raise NotImplementedError
