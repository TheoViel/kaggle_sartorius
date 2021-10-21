import cv2
import numpy as np
from torch.utils.data import Dataset


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

        self.masks = [np.load(path).transpose(1, 2, 0).astype(np.int16) for path in self.mask_paths]

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, idx):
        image = cv2.imread(self.img_paths[idx])
        # masks = np.load(self.mask_paths[idx]).transpose(1, 2, 0).astype(np.int16)
        masks = self.masks[idx]

        if self.transforms:
            transformed = self.transforms(image=image, mask=masks)
            image = transformed["image"]
            masks = transformed["mask"]
            masks = masks.transpose(1, 2).transpose(0, 1).float()

        masks[-1] = masks[-1] / 10000.

        return image, masks
