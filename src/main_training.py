import torch
import argparse
import warnings

from training.main import k_fold
from utils.logger import create_logger, save_config


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
        "--reduce_stride",
        type=bool,
        default=False,
        help="Whether to reduce stride",
    )

    parser.add_argument(
        "--pretrained_folder",
        type=str,
        default=None,
        help="Folder for pretrained weights",
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
    first_epoch_eval = 0
    compute_val_loss = False
    verbose_eval = 5

    device = "cuda" if torch.cuda.is_available() else "cpu"
    save_weights = True

    # Images
    fix = False
    use_mosaic = False
    use_tta = False  # TODO
    # data_config = "data/config_mosaic.py" if use_mosaic else "data/config.py"
    data_config = "data/config.py"

    # k-fold
    k = 5
    random_state = 0
    selected_folds = [0, 1, 2, 3, 4]

    # Model
    name = "maskrcnn"  # "cascade"
    reduce_stride = False
    pretrain = False

    if pretrain and reduce_stride:
        model_config = f"model_zoo/config_{name}_stride_pretrain.py"
    elif pretrain:
        model_config = f"model_zoo/config_{name}_pretrain.py"
    elif reduce_stride:
        model_config = f"model_zoo/config_{name}_stride.py"
    else:
        model_config = f"model_zoo/config_{name}.py"

    pretrained_folder = None
    # pretrained_folder = "../logs/2021-11-04/6/"

    # Training
    optimizer = "Adam"
    scheduler = "plateau" if optimizer == "SGD" else "linear"
    weight_decay = 0.0005 if optimizer == "SGD" else 0
    batch_size = 4 if reduce_stride else 8
    val_bs = batch_size

    epochs = 50

    lr = 5e-4  # 1e-3
    warmup_prop = 0.05

    use_fp16 = False  # TODO


if __name__ == "__main__":
    warnings.simplefilter("ignore", UserWarning)

    args = parse_args()

    config = Config
    config.selected_folds = [args.fold]

    if args.reduce_stride and not config.reduce_stride:
        # Update config to reduced stride
        config.reduce_stride = True
        config.model_config = f"model_zoo/config_{config.name}_stride.py"
        config.batch_size //= 2

    if config.pretrained_folder is None and args.pretrained_folder is not None:
        # Update params
        config.pretrained_folder = args.pretrained_folder
        config.warmup_prop = 0.1
        config.lr /= 2
        config.epochs -= 10
        config.scheduler = "plateau"

    log_folder = args.log_folder

    save_config(Config, log_folder)
    create_logger(directory=log_folder, name="logs.txt")

    k_fold(Config, log_folder=log_folder)
