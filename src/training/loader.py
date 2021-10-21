from torch.utils.data import DataLoader
from utils.torch import worker_init_fn
from params import NUM_WORKERS


def define_loaders(train_dataset, val_dataset,  batch_size=32, val_bs=32):
    """
    Builds data loaders.

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
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        worker_init_fn=worker_init_fn
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=val_bs,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    return train_loader, val_loader
