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
    first_epoch_eval = 10
    compute_val_loss = False
    verbose_eval = 10

    device = "cuda" if torch.cuda.is_available() else "cpu"
    save_weights = True

    # Images
    use_tta = False  # TODO
    data_config = "data/config.py"

    # k-fold
    k = 5
    random_state = 0
    selected_folds = [0, 1, 2, 3, 4]

    # Model
    reduce_stride = False
    model_config = "model_zoo/config.py"
    name = "maskrcnn"
    pretrained_folder = None

    # Training
    optimizer = "Adam"
    weight_decay = 0  # 0.0001
    batch_size = 8
    val_bs = batch_size

    epochs = 50

    lr = 2e-3
    warmup_prop = 0.05

    use_fp16 = False  # TODO


if __name__ == "__main__":
    warnings.simplefilter("ignore", UserWarning)

    args = parse_args()

    config = Config
    config.selected_folds = [args.fold]

    if args.reduce_stride:
        config.reduce_stride = True
        config.model_config = "model_zoo/config_stride.py"
        config.lr /= 2
        config.batch_size //= 2
        config.warmup_prop *= 2

    config.pretrained_folder = args.pretrained_folder

    log_folder = args.log_folder

    save_config(Config, log_folder + "config.json")
    create_logger(directory=log_folder, name="logs.txt")

    k_fold(Config, log_folder=log_folder)
