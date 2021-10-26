from params import SIZE

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True
)

train_viz_pipeline = [
    # dict(type="Resize", img_scale=(1333, 800), keep_ratio=True),
    dict(type="RandomCrop", crop_size=(SIZE, SIZE)),
    dict(type="RandomFlip", flip_ratio=0.5, direction=['horizontal', 'vertical']),
    # dict(type="Mosaic", img_scale=(640, 640)),
    # dict(type="Normalize", **img_norm_cfg),
    dict(type="Pad", size_divisor=32),
    # dict(type="DefaultFormatBundle"),
    dict(
        type="Collect",
        keys=["img", "gt_bboxes", "gt_labels", "gt_masks"],
    )
]

train_pipeline = [
    dict(type="RandomCrop", crop_size=(SIZE, SIZE)),
    dict(type="RandomFlip", flip_ratio=0.5, direction=['horizontal', 'vertical']),
    # dict(type="Mosaic", img_scale=(640, 640)),
    dict(type="Normalize", **img_norm_cfg),
    dict(type="Pad", size_divisor=32),
    dict(type="DefaultFormatBundle"),
    dict(
        type="Collect",
        keys=["img", "gt_bboxes", "gt_labels", "gt_masks"],
    )
]

test_viz_pipeline = [
    # dict(type="Resize", img_scale=(1333, 800), keep_ratio=True),
    dict(type="RandomCrop", crop_size=(SIZE, SIZE)),
    dict(type="RandomFlip", flip_ratio=0.5, direction=['horizontal', 'vertical']),
    # dict(type="Mosaic", img_scale=(640, 640)),
    # dict(type="Normalize", **img_norm_cfg),
    dict(type="Pad", size_divisor=32),
    # dict(type="DefaultFormatBundle"),
    dict(
        type="Collect",
        keys=["img", "gt_bboxes", "gt_labels", "gt_masks"],
    )
]

val_pipeline = [
    dict(type="Pad", size_divisor=32, pad_val=dict(img=(130, 130, 130), masks=0, seg=255)),
    dict(type="Normalize", **img_norm_cfg),
    dict(type="DefaultFormatBundle"),
    dict(
        type="Collect",
        keys=["img", "gt_bboxes", "gt_labels", "gt_masks"],
    )
]

val_viz_pipeline = [
    dict(type="Pad", size_divisor=32, pad_val=dict(img=(130, 130, 130), masks=0, seg=255)),
    dict(
        type="Collect",
        keys=["img", "gt_bboxes", "gt_labels", "gt_masks"],
    )
]

test_pipeline = [
    dict(type="Pad", size_divisor=32, pad_val=dict(img=(130, 130, 130), masks=0, seg=255)),
    dict(type="Normalize", **img_norm_cfg),
    dict(type="DefaultFormatBundle"),
    dict(
        type="Collect",
        keys=["img"],
    )
]

test_viz_pipeline = [
    dict(type="Pad", size_divisor=32, pad_val=dict(img=(130, 130, 130), masks=0, seg=255)),
    dict(
        type="Collect",
        keys=["img"],
    )
]


data = dict(
    train=dict(pipeline=train_pipeline),
    train_viz=dict(pipeline=train_viz_pipeline),
    val=dict(pipeline=val_pipeline),
    val_viz=dict(pipeline=val_viz_pipeline),
    test=dict(pipeline=test_pipeline),
    test_viz=dict(pipeline=test_viz_pipeline),
)
