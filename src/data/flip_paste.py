import copy
import random
import numpy as np
from mmdet.datasets import PIPELINES
from mmdet.datasets.pipelines.transforms import Albu


bbox_params = dict(
    type="BboxParams",
    format="pascal_voc",
    label_fields=["gt_labels"],
    min_visibility=0.25,
    filter_lost_elements=True,
)
keymap = {"img": "image", "gt_masks": "masks", "gt_bboxes": "bboxes"}


@PIPELINES.register_module()
class FlipPaste:
    """
    Adaptation of the CopyPaste augmentation to use an image and its flipped version.
    """
    def __init__(self, p=0.5):
        self.p = p
        self.flip = Albu([dict(type="HorizontalFlip", p=1)], keymap=keymap, bbox_params=bbox_params)

    def __call__(self, input):

        # only apply this transfo for cort (label 2)
        current_label = np.unique(input["gt_labels"])[0]
        if current_label != 2:
            return input

        if random.random() > self.p:
            return input

        aux_input = copy.deepcopy(input)
        flip_image = self.flip(aux_input)

        new_masks = copy.deepcopy(flip_image["gt_masks"])
        new_masks.masks = np.vstack([input["gt_masks"].masks, flip_image["gt_masks"].masks])
        input["gt_masks"] = new_masks

        new_bboxes = np.vstack([np.array(input["gt_bboxes"]), np.array(flip_image["gt_bboxes"])])

        input["gt_bboxes"] = new_bboxes
        input["gt_labels"] = np.hstack([input["gt_labels"], flip_image["gt_labels"]])
        input["gt_semantic_seg"] = np.logical_or(
            input["gt_semantic_seg"], flip_image["gt_semantic_seg"]
        ).astype(np.uint8)
        input["img"] = (0.5 * input["img"] + 0.5 * flip_image["img"]).astype(np.uint8)
        input["scale_factor"] = 1.0

        return input
