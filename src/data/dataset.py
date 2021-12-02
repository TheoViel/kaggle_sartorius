import cv2
# import torch
import pycocotools
import numpy as np
from torch.utils.data import Dataset
from mmdet.core import BitmapMasks

from params import ORIG_SIZE


RESULTS_PH = {
    # 'scale_factor': 1.,  # np.ones(4, dtype=np.float32),  # if no resizing in augs
    "pad_shape": (0, 0),
    "img_norm_cfg": None,
    "flip_direction": None,
    "flip": None,
    'img_fields': ["img"],
    'bbox_fields': ["gt_bboxes"],
    'mask_fields': ["gt_masks"],
    'seg_fields': ['gt_semantic_seg'],
    # 'img_prefix': None,
    # 'img_info': {
    #     'filename': ""
    # }
}


class SartoriusDataset(Dataset):
    """
    Segmentation dataset for training / validation.
    """
    def __init__(self, df, transforms, precompute_masks=True, df_extra=None):
        """
        Constructor.

        Args:
            df (pandas DataFrame): Metadata.
            transforms (MMDet transforms): Augmentation pipeline to apply.
            precompute_masks (bool, optional): Whether to precompute masks. Defaults to True.
            df_extra (pandas DataFrame, optional): Extra data for training. Defaults to None.
        """
        self.df = df
        self.transforms = transforms

        self.img_paths = df["img_path"].values
        self.cell_types = df["cell_type"].values

        self.boxes = [np.array(df['ann'][i]['bboxes']).astype(np.float32) for i in range(len(df))]
        self.class_labels = [np.array(df['ann'][i]['labels']) for i in range(len(df))]

        self.encodings = [np.array(df['ann'][i]['masks']) for i in range(len(df))]

        self.precompute_masks = precompute_masks
        if self.precompute_masks:
            self.masks = [self._load_masks(enc, ORIG_SIZE) for enc in self.encodings]
        else:
            self.masks = None

        self.anns = df['ann'].values

        self.df_extra = df_extra
        self.sample_extra_data(0)

    def sample_extra_data(self, n=None):
        """
        Samples data from the external dataset.
        Use self.n_extra_samples if n is None.

        Args:
            n (int, optional): Number of images to sample. Defaults to None.
        """
        if n is None:
            n = self.n_extra_samples

        self.n_extra_samples = n
        if self.df_extra is None or n == 0:
            return

        df_extra = self.df_extra.sample(self.n_extra_samples).reset_index(drop=True)
        self.img_paths_extra = df_extra["img_path"].values
        self.encodings_extra = [np.array(df_extra['ann'][i]['masks']) for i in range(n)]

        self.boxes_extra = [
            np.array(df_extra['ann'][i]['bboxes']).astype(np.float32) for i in range(n)
        ]
        self.class_labels_extra = [np.array(df_extra['ann'][i]['labels']) for i in range(n)]

    def __len__(self):
        return self.df.shape[0] + self.n_extra_samples

    def __getitem__(self, idx):
        """
        Item accessor.
        Will sample in df_extra if idx > len(df).

        Args:
            idx (int): Sample index.

        Returns:
            dict: Dictionary in the MMDet format.
        """
        if idx < len(self.df):
            path = self.img_paths[idx]
            boxes = self.boxes[idx]
            class_labels = self.class_labels[idx]

            image = cv2.imread(path)

            if self.masks is not None:
                masks = self.masks[idx]
            else:
                masks = self._load_masks(self.encodings[idx], image.shape[:2])
        else:
            idx -= len(self.df)

            path = self.img_paths_extra[idx]
            boxes = self.boxes_extra[idx]
            class_labels = self.class_labels_extra[idx]

            image = cv2.imread(path)
            masks = self._load_masks(self.encodings_extra[idx], image.shape[:2])

        results = {
            "img": image,
            "gt_bboxes": boxes,
            "gt_labels": class_labels,
            "gt_masks": masks,
            'gt_semantic_seg': masks.masks.max(0),
            "img_shape": image.shape[:2],
            "ori_shape": image.shape[:2],
            "filename": path,
            'ori_filename': path,
            "is_extra": int(idx >= len(self.df)),
        }
        results.update(RESULTS_PH)

        results_transfo = None
        while results_transfo is None:
            try:
                results_transfo = self.transforms(results.copy())
            except KeyError:
                results['scale_factor'] = 1.
                results_transfo = self.transforms(results.copy())

        return results_transfo

    def _load_masks(self, mask, shape):
        """
        Loads masks from encodings.

        Args:
            mask (list): Mask encodings.
            shape (tuple): mask shape

        Returns:
            BitmapMasks: Masks in the format adapted to MMDt.
        """
        h, w = shape
        return BitmapMasks([pycocotools.mask.decode(m) for m in mask], h, w)


class SartoriusInferenceDataset(Dataset):
    """
    Segmentation dataset for inference.
    TODO
    """
    def __init__(self, df, transforms, precompute_masks=True):
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

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, idx):
        image = cv2.imread(self.img_paths[idx])

        results = {
            "img": image,
            "img_shape": image.shape[:2],
            "ori_shape": image.shape[:2],
            "filename": self.img_paths[idx],
            'ori_filename': self.img_paths[idx],
            'scale_factor': np.ones(4, dtype=np.float32),
        }
        results.update(RESULTS_PH)
        del results['bbox_fields'], results['mask_fields']

        results_transfo = self.transforms(results.copy())

        # if 'scale_factor' not in results_transfo.keys():
        #     results_transfo['scale_factor'] = np.ones(4)

        return results_transfo
