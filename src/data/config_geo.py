# https://github.com/open-mmlab/mmdetection/blob/master/configs/_base_/datasets/coco_instance.py

from params import SIZE, ORIG_SIZE

H, W = ORIG_SIZE
# W = 1000

img_norm_cfg = dict(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

bbox_params = dict(
    type="BboxParams",
    format="pascal_voc",
    label_fields=["gt_labels"],
    min_visibility=0.25,
    filter_lost_elements=True,
)
keymap = {"img": "image", "gt_masks": "masks", "gt_bboxes": "bboxes"}

crop_rotate_crop = dict(
    type="Compose",
    transforms=[
        dict(type="RandomCrop", height=SIZE * 2, width=SIZE * 2, p=1),
        dict(type="Rotate", limit=45, p=1),
        dict(type="CenterCrop", height=SIZE, width=SIZE, p=1),
    ],
    p=1,
)

crop_resize_crop = dict(
    type="Compose",
    transforms=[
        dict(type="RandomCrop", height=SIZE * 2, width=SIZE * 2, p=1),
        dict(type="RandomScale", scale_limit=(0.75, 1.5), p=1),
        dict(type="RandomCrop", height=SIZE, width=SIZE, p=1),
    ],
    p=1,
)

albu_transforms = [
    dict(type="HorizontalFlip", p=0.5),
    dict(type="VerticalFlip", p=0.5),
    dict(
        type="OneOf",
        transforms=[
            # crop_rotate_crop,
            # dict(type="RandomCrop", height=SIZE, width=SIZE, p=1),
            crop_resize_crop,
        ],
        p=1
    )
]

train_pipeline = [
    dict(
        type="Albu",
        transforms=albu_transforms,
        bbox_params=bbox_params,
        keymap=keymap,
        update_pad_shape=False,
        skip_img_without_anno=True,
    ),
    dict(type="Normalize", **img_norm_cfg),
    dict(type="Pad", size_divisor=32, pad_val=dict(img=(130, 130, 130), masks=0, seg=255)),
    dict(type="DefaultFormatBundle"),
    dict(
        type="Collect",
        keys=["img", "gt_bboxes", "gt_labels", "gt_masks"],
    ),
]

val_pipeline = [
    # dict(type="Resize", img_scale=(W, W), keep_ratio=True),
    dict(type="Pad", size_divisor=32, pad_val=dict(img=(130, 130, 130), masks=0, seg=255)),
    dict(type="Normalize", **img_norm_cfg),
    dict(type="DefaultFormatBundle"),
    dict(
        type="Collect",
        keys=["img", "gt_bboxes", "gt_labels", "gt_masks"],
    ),
]

test_pipeline = [
    # dict(type="Resize", img_scale=(W, W), keep_ratio=True),
    dict(type="Pad", size_divisor=32, pad_val=dict(img=(130, 130, 130), masks=0, seg=255)),
    dict(type="Normalize", **img_norm_cfg),
    dict(type="ImageToTensor", keys=["img"]),
    dict(type="Collect", keys=["img"]),
]

train_viz_pipeline = train_pipeline[:-4] + [train_pipeline[-1]]
val_viz_pipeline = [val_pipeline[-1]]
test_viz_pipeline = [test_pipeline[-1]]

data = dict(
    train=dict(pipeline=train_pipeline),
    train_viz=dict(pipeline=train_viz_pipeline),
    val=dict(pipeline=val_pipeline),
    val_viz=dict(pipeline=val_viz_pipeline),
    test=dict(pipeline=test_pipeline),
    test_viz=dict(pipeline=test_viz_pipeline),
)
