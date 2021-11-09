# Adaptation of the Mosaic class to include masks.
# The function is not actually used but saved here as backup.
# It is actually in the site-packages/mmdet/datasets/pipelines/transforms.py file.

import copy
import random
import numpy as np
from mmdet.core import BitmapMasks


class Mosaic:
    """Mosaic augmentation.

    Given 4 images, mosaic transform combines them into
    one output image. The output image is composed of the parts from each sub-
    image.

    .. code:: text

                        mosaic transform
                           center_x
                +------------------------------+
                |       pad        |  pad      |
                |      +-----------+           |
                |      |           |           |
                |      |  image1   |--------+  |
                |      |           |        |  |
                |      |           | image2 |  |
     center_y   |----+-------------+-----------|
                |    |   cropped   |           |
                |pad |   image3    |  image4   |
                |    |             |           |
                +----|-------------+-----------+
                     |             |
                     +-------------+

     The mosaic transform steps are as follows:

         1. Choose the mosaic center as the intersections of 4 images
         2. Get the left top image according to the index, and randomly
            sample another 3 images from the custom dataset.
         3. Sub image will be cropped if image is larger than mosaic patch

    Args:
        img_scale (Sequence[int]): Image size after mosaic pipeline of single
           image. Default to (640, 640).
        center_ratio_range (Sequence[float]): Center ratio range of mosaic
           output. Default to (0.5, 1.5).
        min_bbox_size (int | float): The minimum pixel for filtering
            invalid bboxes after the mosaic pipeline. Default to 0.
        pad_val (int): Pad value. Default to 114.
    """

    def __init__(
        self,
        img_scale=(640, 640),
        center_ratio_range=(0.5, 1.5),
        min_bbox_size=0,
        pad_val=114,
        p=1,
    ):
        assert isinstance(img_scale, tuple)
        self.img_scale = img_scale
        self.center_ratio_range = center_ratio_range
        self.min_bbox_size = min_bbox_size
        self.pad_val = pad_val
        self.p = p

    def __call__(self, results):
        """Call function to make a mosaic of image.

        Args:
            results (dict): Result dict.

        Returns:
            dict: Result dict with mosaic transformed.
        """
        if np.random.random() < self.p:
            results = self._mosaic_transform(results)

        return results

    def get_indexes(self, dataset):
        """Call function to collect indexes.

        Args:
            dataset (:obj:`MultiImageMixDataset`): The dataset.

        Returns:
            list: indexes.
        """

        indexs = [random.randint(0, len(dataset)) for _ in range(3)]
        return indexs

    def _mosaic_transform(self, results):
        """Mosaic transform function.

        Args:
            results (dict): Result dict.

        Returns:
            dict: Updated result dict.
        """

        assert "mix_results" in results
        mosaic_labels = []
        mosaic_bboxes = []
        mosaic_masks = []

        if len(results["img"].shape) == 3:
            mosaic_img = np.full(
                (int(self.img_scale[0] * 2), int(self.img_scale[1] * 2), 3),
                self.pad_val,
                dtype=results["img"].dtype,
            )
        else:
            mosaic_img = np.full(
                (int(self.img_scale[0] * 2), int(self.img_scale[1] * 2)),
                self.pad_val,
                dtype=results["img"].dtype,
            )

        # mosaic center x, y
        center_x = int(random.uniform(*self.center_ratio_range) * self.img_scale[1])
        center_y = int(random.uniform(*self.center_ratio_range) * self.img_scale[0])
        center_position = (center_x, center_y)

        loc_strs = ("top_left", "top_right", "bottom_left", "bottom_right")
        for i, loc in enumerate(loc_strs):
            if loc == "top_left":
                results_patch = copy.deepcopy(results)
            else:
                results_patch = copy.deepcopy(results["mix_results"][i - 1])

            img_i = results_patch["img"]
            h_i, w_i = img_i.shape[:2]
            # keep_ratio resize
            scale_ratio_i = min(self.img_scale[0] / h_i, self.img_scale[1] / w_i)

            assert scale_ratio_i == 1.0, "Resizing not supported"

            # compute the combine parameters
            paste_coord, crop_coord = self._mosaic_combine(
                loc, center_position, img_i.shape[:2][::-1]
            )
            x1_p, y1_p, x2_p, y2_p = paste_coord
            x1_c, y1_c, x2_c, y2_c = crop_coord

            # crop and paste image
            mosaic_img[y1_p:y2_p, x1_p:x2_p] = img_i[y1_c:y2_c, x1_c:x2_c]

            # crop and paste masks

            mosaic_mask = np.zeros(
                (
                    len(results_patch["gt_masks"]),
                    mosaic_img.shape[0],
                    mosaic_img.shape[1],
                ),
                dtype=results_patch["gt_masks"].masks.dtype,
            )

            mosaic_mask[:, y1_p:y2_p, x1_p:x2_p] = results_patch["gt_masks"].masks[
                :, y1_c:y2_c, x1_c:x2_c
            ]
            mosaic_masks.append(mosaic_mask)

            # adjust coordinate
            gt_bboxes_i = results_patch["gt_bboxes"]
            gt_labels_i = results_patch["gt_labels"]

            if gt_bboxes_i.shape[0] > 0:
                padw = x1_p - x1_c
                padh = y1_p - y1_c
                gt_bboxes_i[:, 0::2] = scale_ratio_i * gt_bboxes_i[:, 0::2] + padw
                gt_bboxes_i[:, 1::2] = scale_ratio_i * gt_bboxes_i[:, 1::2] + padh

            mosaic_bboxes.append(gt_bboxes_i)
            mosaic_labels.append(gt_labels_i)

        mosaic_masks = np.concatenate(mosaic_masks, 0)

        if len(mosaic_labels) > 0:
            mosaic_bboxes = np.concatenate(mosaic_bboxes, 0)
            mosaic_bboxes[:, 0::2] = np.clip(
                mosaic_bboxes[:, 0::2], 0, 2 * self.img_scale[1]
            )
            mosaic_bboxes[:, 1::2] = np.clip(
                mosaic_bboxes[:, 1::2], 0, 2 * self.img_scale[0]
            )
            mosaic_labels = np.concatenate(mosaic_labels, 0)

            mosaic_bboxes, mosaic_labels, mosaic_masks = self._filter_box_candidates(
                mosaic_bboxes, mosaic_labels, mosaic_masks
            )

        results["img"] = mosaic_img
        results["img_shape"] = mosaic_img.shape
        results["ori_shape"] = mosaic_img.shape
        results["gt_bboxes"] = mosaic_bboxes
        results["gt_labels"] = mosaic_labels

        results["gt_masks"] = BitmapMasks(
            mosaic_masks, mosaic_img.shape[0], mosaic_img.shape[1]
        )

        return results

    def _mosaic_combine(self, loc, center_position_xy, img_shape_wh):
        """Calculate global coordinate of mosaic image and local coordinate of
        cropped sub-image.

        Args:
            loc (str): Index for the sub-image, loc in ('top_left',
              'top_right', 'bottom_left', 'bottom_right').
            center_position_xy (Sequence[float]): Mixing center for 4 images,
                (x, y).
            img_shape_wh (Sequence[int]): Width and height of sub-image

        Returns:
            tuple[tuple[float]]: Corresponding coordinate of pasting and
                cropping
                - paste_coord (tuple): paste corner coordinate in mosaic image.
                - crop_coord (tuple): crop corner coordinate in mosaic image.
        """

        assert loc in ("top_left", "top_right", "bottom_left", "bottom_right")
        if loc == "top_left":
            # index0 to top left part of image
            x1, y1, x2, y2 = (
                max(center_position_xy[0] - img_shape_wh[0], 0),
                max(center_position_xy[1] - img_shape_wh[1], 0),
                center_position_xy[0],
                center_position_xy[1],
            )
            crop_coord = (
                img_shape_wh[0] - (x2 - x1),
                img_shape_wh[1] - (y2 - y1),
                img_shape_wh[0],
                img_shape_wh[1],
            )

        elif loc == "top_right":
            # index1 to top right part of image
            x1, y1, x2, y2 = (
                center_position_xy[0],
                max(center_position_xy[1] - img_shape_wh[1], 0),
                min(center_position_xy[0] + img_shape_wh[0], self.img_scale[1] * 2),
                center_position_xy[1],
            )
            crop_coord = (
                0,
                img_shape_wh[1] - (y2 - y1),
                min(img_shape_wh[0], x2 - x1),
                img_shape_wh[1],
            )

        elif loc == "bottom_left":
            # index2 to bottom left part of image
            x1, y1, x2, y2 = (
                max(center_position_xy[0] - img_shape_wh[0], 0),
                center_position_xy[1],
                center_position_xy[0],
                min(self.img_scale[0] * 2, center_position_xy[1] + img_shape_wh[1]),
            )
            crop_coord = (
                img_shape_wh[0] - (x2 - x1),
                0,
                img_shape_wh[0],
                min(y2 - y1, img_shape_wh[1]),
            )

        else:
            # index3 to bottom right part of image
            x1, y1, x2, y2 = (
                center_position_xy[0],
                center_position_xy[1],
                min(center_position_xy[0] + img_shape_wh[0], self.img_scale[1] * 2),
                min(self.img_scale[0] * 2, center_position_xy[1] + img_shape_wh[1]),
            )
            crop_coord = (
                0,
                0,
                min(img_shape_wh[0], x2 - x1),
                min(y2 - y1, img_shape_wh[1]),
            )

        paste_coord = x1, y1, x2, y2
        return paste_coord, crop_coord

    def _filter_box_candidates(self, bboxes, labels, masks):
        """Filter out bboxes too small after Mosaic."""
        bbox_w = bboxes[:, 2] - bboxes[:, 0]
        bbox_h = bboxes[:, 3] - bboxes[:, 1]
        valid_inds = (bbox_w > self.min_bbox_size) & (bbox_h > self.min_bbox_size)
        valid_inds = np.nonzero(valid_inds)[0]
        return bboxes[valid_inds], labels[valid_inds], masks[valid_inds]

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f"img_scale={self.img_scale}, "
        repr_str += f"center_ratio_range={self.center_ratio_range})"
        repr_str += f"pad_val={self.pad_val})"
        return repr_str
