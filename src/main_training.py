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
        "efficientnet_b4": 4,
        "efficientnet_b5": 2,
    },
    "cascade_resnest": {
        "resnest50": 4,
        "resnest101": 3,
    },
    "cascade_mask_scoring": {
        "resnet50": 4,
        "resnext101": 3,
        # "resnext101_64x4": 2,
        # "swin_tiny": 4,
        # "swin_small": 3,
        # "swin_base": 2,
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
        "--encoder",
        type=str,
        default=None,
        help="Encoder",
    )

    parser.add_argument(
        "--name",
        type=str,
        default=None,
        help="Model name",
    )

    parser.add_argument(
        "--use_extra_samples",
        type=bool,
        default=None,
        help="Whether to use extra samples from the Livecell dataset.",
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
    first_epoch_eval = 10
    compute_val_loss = False
    verbose_eval = 10

    device = "cuda" if torch.cuda.is_available() else "cpu"
    save_weights = True

    # Images
    fix = True
    remove_anomalies = True
    extra_name = "livecell_no_shsy5y"
    use_extra_samples = False
    num_classes = 3

    use_mosaic = False
    data_config = "configs/config_aug_mosaic.py" if use_mosaic else "configs/config_aug.py"

    # k-fold
    split = "sgkf"
    k = 5
    random_state = 0
    selected_folds = [0]

    # Model
    name = "cascade"  # "cascade" "maskrcnn" "htc"
    encoder = "efficientnet_b5"
    model_config = f"configs/config_{name}.py"
    pretrained_livecell = True
    freeze_bn = True

    if name == "htc":
        data_config = "configs/config_aug_semantic.py"

    # Training
    optimizer = "AdamW"
    scheduler = "linear"
    weight_decay = 0.01 if optimizer == "AdamW" else 0
    batch_size = BATCH_SIZES[name][encoder]
    val_bs = batch_size

    epochs = 10 * batch_size

    lr = 2e-4
    warmup_prop = 0.05


if __name__ == "__main__":
    warnings.simplefilter("ignore", UserWarning)

    args = parse_args()

    config = Config
    config.selected_folds = [args.fold]

    if args.epochs is not None:
        config.epochs = args.epochs

    if args.lr is not None:
        config.lr = args.lr

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

    if args.use_extra_samples is not None:
        config.use_extra_samples = args.use_extra_samples

    log_folder = args.log_folder

    save_config(Config, log_folder)
    create_logger(directory=log_folder, name="logs.txt")

    k_fold(Config, log_folder=log_folder)
