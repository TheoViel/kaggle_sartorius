import cv2
# import torch
import pycocotools
import numpy as np
from torch.utils.data import Dataset
from mmdet.core import BitmapMasks

from params import CELL_TYPES, ORIG_SIZE


RESULTS_PH = {
    'scale_factor': 1.,  # np.ones(4),
    "pad_shape": (0, 0),
    "img_norm_cfg": None,
    "flip_direction": None,
    "flip": None,
    'img_fields': ["img"],
    'bbox_fields': ["gt_bboxes"],
    'mask_fields': ["gt_masks"]
}


class SartoriusDataset(Dataset):
    """
    Segmentation dataset for training / validation.
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
        self.cell_types = df["cell_type"].values

        self.y_cls = [CELL_TYPES.index(c) for c in self.cell_types]

        self.boxes = [np.array(df['ann'][i]['bboxes']).astype(np.float32) for i in range(len(df))]
        self.class_labels = [np.array(df['ann'][i]['labels']) for i in range(len(df))]

        self.encodings = [np.array(df['ann'][i]['masks']) for i in range(len(df))]

        if precompute_masks:
            self.masks = [self._load_masks(enc, ORIG_SIZE) for enc in self.encodings]
        else:
            self.masks = None

        self.anns = df['ann'].values

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, idx):
        image = cv2.imread(self.img_paths[idx])

        if self.masks is not None:
            masks = self.masks[idx]
        else:
            masks = self._load_masks(self.encodings[idx], image.shape[:2])

        results = {
            "img": image,
            "gt_bboxes": self.boxes[idx],
            "gt_labels": self.class_labels[idx] * 0,
            "gt_masks": masks,
            "img_shape": image.shape[:2],
            "ori_shape": image.shape[:2],
            "filename": self.img_paths[idx],
            'ori_filename': self.img_paths[idx],
        }
        results.update(RESULTS_PH)

        results_transfo = None
        while results_transfo is None:
            results_transfo = self.transforms(results.copy())

        # if 'scale_factor' not in results_transfo.keys():
        #     results_transfo['scale_factor'] = np.ones(4)

        return results_transfo

    def _load_masks(self, mask, shape):
        h, w = shape
        return BitmapMasks([pycocotools.mask.decode(m) for m in mask], h, w)
