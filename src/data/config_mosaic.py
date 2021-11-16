# https://github.com/open-mmlab/mmdetection/blob/master/configs/_base_/datasets/coco_instance.py

SIZE = 256
SIZE_MOSAIC = int(0.75 * SIZE)

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True
)

train_pipeline = [
    dict(type="RandomCrop", crop_size=(SIZE_MOSAIC, SIZE_MOSAIC)),
    dict(type="RandomFlip", flip_ratio=0.5, direction=['horizontal', 'vertical']),
]

mosaic_pipeline = [
    dict(
        type="Mosaic",
        img_scale=(SIZE_MOSAIC, SIZE_MOSAIC),
        center_ratio_range=(1., 1.),
        min_bbox_size=5,
        p=1,  # lower not really supported, I need to find a hack
    ),
    dict(type="RandomCrop", crop_size=(SIZE, SIZE)),
    dict(type="Normalize", **img_norm_cfg),
    dict(type="Pad", size_divisor=32, pad_val=dict(img=(130, 130, 130), masks=0, seg=255)),
    dict(type="DefaultFormatBundle"),
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

test_pipeline = [
    dict(
        type='MultiScaleFlipAug',
        img_scale=[(520, 704)],
        flip=False,
        transforms=[
            dict(type="Pad", size_divisor=32, pad_val=dict(img=(130, 130, 130), masks=0, seg=255)),
            dict(type="Normalize", **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type="Collect", keys=["img"])
        ]
    )
]

test_pipeline_tta = [
    dict(
        type='MultiScaleFlipAug',
        img_scale=[(520, 704)],
        flip=True,
        flip_direction=['horizontal', 'vertical', "diagonal"],
        transforms=[
            dict(type='RandomFlip'),
            dict(type="Pad", size_divisor=32, pad_val=dict(img=(130, 130, 130), masks=0, seg=255)),
            dict(type="Normalize", **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type="Collect", keys=["img"])
        ]
    )
]

mosaic_viz_pipeline = mosaic_pipeline[:-4] + [mosaic_pipeline[-1]]
val_viz_pipeline = [val_pipeline[-1]]
test_viz_pipeline = [test_pipeline[-1]]


data = dict(
    train=dict(pipeline=train_pipeline),
    mosaic=dict(pipeline=mosaic_pipeline),
    mosaic_viz=dict(pipeline=mosaic_viz_pipeline),
    val=dict(pipeline=val_pipeline),
    val_viz=dict(pipeline=val_viz_pipeline),
    test=dict(pipeline=test_pipeline),
    test_tta=dict(pipeline=test_pipeline_tta),
    test_viz=dict(pipeline=test_viz_pipeline),
)
