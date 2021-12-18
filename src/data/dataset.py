import cv2
import torch
from torch.utils.data import Dataset

from params import CELL_TYPES


class SartoriusDataset(Dataset):
    """
    Segmentation dataset for training / validation.
    """
    def __init__(self, df, transforms=None):
        """
        Constructor.
        Args:
            df (pandas dataframe): Metadata.
            transforms (albumentation transforms, optional): Transforms to apply. Defaults to None.
            train (bool, optional): Indicates if the dataset is used for training. Defaults to True.
        """

        self.df = df
        self.transforms = transforms

        self.img_paths = df["img_path"].values

        self.cell_types = df["cell_type"].values
        self.plates = df["plate_class"].values

        self.y_cell = [CELL_TYPES.index(c) for c in self.cell_types]
        self.y_plate = df["plate_class"].values

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, idx):
        image = cv2.imread(self.img_paths[idx])

        if self.transforms:
            transformed = self.transforms(image=image)
            image = transformed["image"]

        y_cell = torch.tensor(self.y_cell[idx]).long()
        y_plate = torch.tensor(self.y_plate[idx]).long()

        return image, y_cell, y_plate
