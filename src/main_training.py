import torch
import argparse
import warnings

from training.main import k_fold
from utils.logger import create_logger, save_config

BATCH_SIZES = {
    "maskrcnn": {
        "resnet50": 4,
        "resnet101": 4,
        "resnext101": 4,
        "efficientnet_b4": 4,
        "efficientnet_b5": 3,
        "efficientnet_b6": 2,
    },
    "cascade": {
        "resnet50": 4,
        "resnext101": 3,
        "resnext101_64x4": 2,
        "swin_tiny": 4,
        "swin_small": 3,
        "swin_base": 2,
        "efficientnetv2_s": 4,
        "efficientnetv2_m": 3,
        "efficientnet_b4": 3,
        "efficientnet_b5": 2,
        "efficientnet_b6": 2,
    },
    "cascade_resnest": {
        "resnest50": 4,
        "resnest101": 3,
    },
    "cascade_mask_scoring": {
        "resnet50": 4,
        "resnext101": 3,
    },
    "htc": {
        "resnet50": 3,
        "resnext101": 2,
    },
}


def parse_args():
    """
    Parses arguments
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--fold",
        type=int,
        default=0,
        help="Fold number",
    )

    parser.add_argument(
        "--log_folder",
        type=str,
        default="",
        help="Folder to log results to",
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        help="Learning rate",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Epochs",
    )

    parser.add_argument(
        "--name",
        type=str,
        default=None,
        help="Model name",
    )

    parser.add_argument(
        "--encoder",
        type=str,
        default=None,
        help="Encoder",
    )

    parser.add_argument(
        "--freeze_bn",
        type=int,
        default=None,
        help="Whether to freeze batch norm layers.",
    )

    args = parser.parse_args()

    return args


class Config:
    """
    Parameters used for training
    """

    # General
    seed = 42
    verbose = 1
    first_epoch_eval = 5
    compute_val_loss = False
    verbose_eval = 5

    device = "cuda" if torch.cuda.is_available() else "cpu"
    save_weights = True

    # Images
    fix = True
    remove_anomalies = True

    extra_name = "livecell_no_shsy5y"
    use_extra_samples = False
    use_pl = True

    num_classes = 3

    data_config = "configs/config_aug.py"
    # data_config = "configs/config_aug_extra.py"  # flip_paste

    # k-fold
    split = "gkf"
    k = 5
    random_state = 0
    selected_folds = [0, 1, 2, 3, 4]

    # Model
    name = "cascade"  # "cascade" "maskrcnn"
    encoder = "resnext101"
    model_config = f"configs/config_{name}.py"
    pretrained_livecell = True
    freeze_bn = False  # True ?

    if name == "htc":
        data_config = "configs/config_aug_semantic.py"

    # Training
    optimizer = "AdamW"
    scheduler = "linear"
    weight_decay = 0.01 if optimizer == "AdamW" else 0
    batch_size = BATCH_SIZES[name][encoder]
    val_bs = batch_size
    loss_decay = True

    epochs = 10 * batch_size
    if use_pl or use_extra_samples:
        epochs = epochs // 2

    lr = 2e-4
    warmup_prop = 0.05


if __name__ == "__main__":
    warnings.simplefilter("ignore", UserWarning)

    args = parse_args()

    config = Config
    config.selected_folds = [args.fold]

    if args.lr is not None:
        config.lr = args.lr

    if args.freeze_bn is not None:
        config.freeze_bn = bool(args.freeze_bn)

    if args.encoder is not None:
        config.encoder = args.encoder

    if args.name is not None:
        config.name = args.name

        config.model_config = f"configs/config_{config.name}.py"
        if config.name == "htc":
            config.data_config = "configs/config_aug_semantic.py"

    if args.encoder is not None or args.name is not None:
        config.batch_size = BATCH_SIZES[config.name][config.encoder]
        config.epochs = 10 * config.batch_size

    if args.epochs is not None:
        config.epochs = args.epochs

    log_folder = args.log_folder

    save_config(Config, log_folder)
    create_logger(directory=log_folder, name="logs.txt")

    k_fold(Config, log_folder=log_folder)
