# https://github.com/open-mmlab/mmdetection/blob/master/configs/_base_/datasets/coco_instance.py

SIZE = 256
# SIZE = 384

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True
)

train_pipeline = [
    dict(type="RandomCrop", crop_size=(SIZE, SIZE)),
    dict(type="RandomFlip", flip_ratio=0.5, direction=['horizontal', 'vertical']),
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
    dict(type="Pad", size_divisor=32, pad_val=dict(img=(130, 130, 130), masks=0, seg=255)),
    dict(type="Normalize", **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type="Collect", keys=["img"])
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
