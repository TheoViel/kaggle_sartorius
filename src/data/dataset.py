import cv2
import torch
import numpy as np
from torch.utils.data import Dataset

from params import CELL_TYPES


class SartoriusDataset(Dataset):
    """
    Segmentation dataset for training / validation.
    """
    def __init__(self, df, transforms=None, train=True):
        """
        Constructor.

        Args:
            df (pandas dataframe): Metadata.
            transforms (albumentation transforms, optional): Transforms to apply. Defaults to None.
            train (bool, optional): Indicates if the dataset is used for training. Defaults to True.
        """

        self.df = df
        self.train = train
        self.transforms = transforms

        self.img_paths = df["img_path"].values
        self.mask_paths = df["mask_path"].values

        self.masks = [np.load(path).transpose(1, 2, 0) for path in self.mask_paths]
        self.cell_types = df["cell_type"].values

        self.y_cls = [CELL_TYPES.index(c) for c in self.cell_types]

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, idx):
        image = cv2.imread(self.img_paths[idx])
        masks = self.masks[idx][..., [0, 2, 3]]

        if self.transforms:
            transformed = self.transforms(image=image, mask=masks)
            image = transformed["image"]
            masks = transformed["mask"]
            masks = masks.transpose(1, 2).transpose(0, 1).float()

        y = torch.tensor(self.y_cls[idx])

        return image, masks, y
