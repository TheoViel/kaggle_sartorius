from functools import partial
from mmcv.parallel import collate
from torch.utils.data import DataLoader

from utils.torch import worker_init_fn
from params import NUM_WORKERS


def define_loaders(
    train_dataset=None, val_dataset=None,  batch_size=32, val_bs=32, num_workers=NUM_WORKERS
):
    """
    Builds data loaders. TODO

    Args:
        train_dataset (CollageingDataset): Dataset to train with.
        val_dataset (CollageingDataset): Dataset to validate with.
        samples_per_patient (int, optional): Number of images to use per patient. Defaults to 0.
        batch_size (int, optional): Training batch size. Defaults to 32.
        val_bs (int, optional): Validation batch size. Defaults to 32.

    Returns:
       DataLoader: Train loader.
       DataLoader: Val loader.
    """
    train_loader, val_loader = None, None

    if train_dataset is not None:
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=num_workers,
            collate_fn=partial(collate, samples_per_gpu=batch_size),
            pin_memory=False,
            worker_init_fn=worker_init_fn
        )

    if val_dataset is not None:
        val_loader = DataLoader(
            val_dataset,
            batch_size=val_bs,
            shuffle=False,
            collate_fn=partial(collate, samples_per_gpu=batch_size),
            num_workers=num_workers,
            pin_memory=True,
        )

    return train_loader, val_loader
