import torch
import numpy as np
from torch.utils.data import Dataset


class SartoriusDataset(Dataset):
    """
    Segmentation dataset for training / validation.
    """
    def __init__(self, paths, transforms=None):
        """
        Constructor.

        Args:
            df (pandas dataframe): Metadata.
            transforms (albumentation transforms, optional): Transforms to apply. Defaults to None.
            train (bool, optional): Indicates if the dataset is used for training. Defaults to True.
        """
        self.paths = paths
        self.transforms = transforms

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        array = np.load(self.paths[idx])
        image = array[:3].transpose(1, 2, 0)
        mask = array[-1]

        y = torch.tensor(mask.max() > 0).float()

        if self.transforms:
            transformed = self.transforms(image=image, mask=mask)
            image = transformed["image"]
            mask = transformed["mask"]
            mask = mask.unsqueeze(0).float()

        return image, mask, y
