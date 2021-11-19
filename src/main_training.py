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
    first_epoch_eval = 1
    compute_val_loss = False
    verbose_eval = 5

    device = "cuda" if torch.cuda.is_available() else "cpu"
    save_weights = True

    # Images
    fix = True
    extra_name = "livecell_no_shsy5y"
    use_extra_samples = False
    num_classes = 3
    pretrained_livecell = True

    use_mosaic = False
    data_config = "data/config_mosaic.py" if use_mosaic else "data/config.py"

    # k-fold
    k = 5
    random_state = 0
    selected_folds = [0]  # , 1, 2, 3, 4]

    # Model
    name = "maskrcnn"  # "cascade" "maskrcnn"
    encoder = "resnet50"
    model_config = f"model_zoo/config_{name}.py"
    pretrained_livecell = True

    # Training
    optimizer = "Adam"
    scheduler = "linear"  # "plateau" "linear"
    weight_decay = 0.0005 if optimizer == "SGD" else 0
    batch_size = 4
    val_bs = batch_size

    epochs = 30 if use_extra_samples else 40

    lr = 3e-4
    warmup_prop = 0.05

    use_fp16 = False  # TODO


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

    if args.use_extra_samples is not None:
        config.use_extra_samples = args.use_extra_samples

    log_folder = args.log_folder

    save_config(Config, log_folder)
    create_logger(directory=log_folder, name="logs.txt")

    k_fold(Config, log_folder=log_folder)
