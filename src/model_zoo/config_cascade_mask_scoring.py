# https://github.com/open-mmlab/mmdetection/blob/master/configs/_base_/models/mask_rcnn_r50_fpn.py

custom_imports = dict(
    imports=['model_zoo.cascade_mask_scoring']
)  # does this work ?

num_classes = 3  # 8
mask_iou_threshold = 0.3
bbox_iou_threshold = 0.7
rpn_thresholds = (0.7, 0.3, 0.3)

pretrained_weights = {  # TODO : Adapt load weights to ignore mask scoring head.
    "resnet50": "../input/weights/cascade_mask_rcnn_r50_fpn_mstrain_3x_coco.pth",
    "resnet101": "../input/weights/cascade_mask_rcnn_r101_fpn_mstrain_3x_coco.pth",
    "resnext101": "../input/weights/cascade_mask_rcnn_x101_32x4d_fpn_mstrain_3x_coco.pth",
    "swin_tiny": "../input/weights/cascade_mask_rcnn_swin_tiny_patch4_window7.pth",
    "swin_small": "../input/weights/cascade_mask_rcnn_swin_small_patch4_window7.pth",
    "swin_base": "../input/weights/cascade_mask_rcnn_swin_base_patch4_window7.pth",
}

pretrained_weights_livecell = {
    "resnet50": "../logs/pretrain/2021-11-21/1/cascade_resnet50_0.pt",  # TODO
    "resnext101": "../logs/pretrain/2021-11-21/0/cascade_resnext101_0.pt",  # TODO
    "swin_tiny": pretrained_weights["swin_tiny"],  # TODO
    "swin_small": pretrained_weights["swin_small"],  # TODO
    "swin_base": pretrained_weights["swin_base"],  # TODO
}


model = dict(
    type="CascadeMaskScoringRCNN",
    backbone="",
    neck=dict(
        type="FPN", in_channels=[256, 512, 1024, 2048], out_channels=256, num_outs=5
    ),
    rpn_head=dict(
        type="RPNHead",
        in_channels=256,
        feat_channels=256,
        anchor_generator=dict(
            type="AnchorGenerator",
            scales=[8],
            ratios=[0.5, 1.0, 2.0],
            strides=[2, 4, 8, 16, 32],
        ),
        bbox_coder=dict(
            type="DeltaXYWHBBoxCoder",
            target_means=[0.0, 0.0, 0.0, 0.0],
            target_stds=[1.0, 1.0, 1.0, 1.0],
        ),
        loss_cls=dict(type="CrossEntropyLoss", use_sigmoid=True, loss_weight=1.0),
        loss_bbox=dict(type="SmoothL1Loss", beta=1.0 / 9.0, loss_weight=1.0),
    ),
    roi_head=dict(
        type="CascadeMaskScoringRoIHead",
        mask_iou_head=dict(
            type='MaskIoUHead',
            num_convs=4,
            num_fcs=1,
            roi_feat_size=14,
            in_channels=256,
            conv_out_channels=256,
            fc_out_channels=512,
            num_classes=num_classes,
        ),
        num_stages=3,
        stage_loss_weights=[1, 0.5, 0.25],
        bbox_roi_extractor=dict(
            type="SingleRoIExtractor",
            roi_layer=dict(type="RoIAlign", output_size=7, sampling_ratio=0),
            out_channels=256,
            featmap_strides=[2, 4, 8, 16],
        ),
        bbox_head=[
            dict(
                type="Shared2FCBBoxHead",
                in_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=num_classes,
                bbox_coder=dict(
                    type="DeltaXYWHBBoxCoder",
                    target_means=[0.0, 0.0, 0.0, 0.0],
                    target_stds=[0.1, 0.1, 0.2, 0.2],
                ),
                reg_class_agnostic=True,
                loss_cls=dict(
                    type="CrossEntropyLoss", use_sigmoid=False, loss_weight=1.0
                ),
                loss_bbox=dict(type="SmoothL1Loss", beta=1.0, loss_weight=1.0),
            ),
            dict(
                type="Shared2FCBBoxHead",
                in_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=num_classes,
                bbox_coder=dict(
                    type="DeltaXYWHBBoxCoder",
                    target_means=[0.0, 0.0, 0.0, 0.0],
                    target_stds=[0.05, 0.05, 0.1, 0.1],
                ),
                reg_class_agnostic=True,
                loss_cls=dict(
                    type="CrossEntropyLoss", use_sigmoid=False, loss_weight=1.0
                ),
                loss_bbox=dict(type="SmoothL1Loss", beta=1.0, loss_weight=1.0),
            ),
            dict(
                type="Shared2FCBBoxHead",
                in_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=num_classes,
                bbox_coder=dict(
                    type="DeltaXYWHBBoxCoder",
                    target_means=[0.0, 0.0, 0.0, 0.0],
                    target_stds=[0.033, 0.033, 0.067, 0.067],
                ),
                reg_class_agnostic=True,
                loss_cls=dict(
                    type="CrossEntropyLoss", use_sigmoid=False, loss_weight=1.0
                ),
                loss_bbox=dict(type="SmoothL1Loss", beta=1.0, loss_weight=1.0),
            ),
        ],
        mask_roi_extractor=dict(
            type="SingleRoIExtractor",
            roi_layer=dict(type="RoIAlign", output_size=14, sampling_ratio=0),
            out_channels=256,
            featmap_strides=[2, 4, 8, 16],
        ),
        mask_head=dict(
            type="FCNMaskHead",
            num_convs=4,
            in_channels=256,
            conv_out_channels=256,
            num_classes=num_classes,
            loss_mask=dict(type="CrossEntropyLoss", use_mask=True, loss_weight=1.0),
        ),
    ),
    # model training and testing settings
    train_cfg=dict(
        rpn=dict(
            assigner=dict(
                type="MaxIoUAssigner",
                pos_iou_thr=rpn_thresholds[0],
                neg_iou_thr=rpn_thresholds[1],
                min_pos_iou=rpn_thresholds[2],
                match_low_quality=True,
                ignore_iof_thr=-1,
            ),
            sampler=dict(
                type="RandomSampler",
                num=256,
                pos_fraction=0.5,
                neg_pos_ub=-1,
                add_gt_as_proposals=False,
            ),
            allowed_border=-1,
            pos_weight=-1,
            debug=False,
        ),
        rpn_proposal=dict(
            nms_pre=2000,
            max_per_img=2000,
            nms=dict(type="nms", iou_threshold=bbox_iou_threshold),
            min_bbox_size=0,
        ),
        rcnn=[
            dict(
                assigner=dict(
                    type="MaxIoUAssigner",
                    pos_iou_thr=0.5,
                    neg_iou_thr=0.5,
                    min_pos_iou=0.5,
                    match_low_quality=False,
                    ignore_iof_thr=-1,
                ),
                sampler=dict(
                    type="RandomSampler",
                    num=512,
                    pos_fraction=0.25,
                    neg_pos_ub=-1,
                    add_gt_as_proposals=True,
                ),
                mask_size=28,
                pos_weight=-1,
                debug=False,
                mask_thr_binary=0.45,
            ),
            dict(
                assigner=dict(
                    type="MaxIoUAssigner",
                    pos_iou_thr=0.6,
                    neg_iou_thr=0.6,
                    min_pos_iou=0.6,
                    match_low_quality=False,
                    ignore_iof_thr=-1,
                ),
                sampler=dict(
                    type="RandomSampler",
                    num=512,
                    pos_fraction=0.25,
                    neg_pos_ub=-1,
                    add_gt_as_proposals=True,
                ),
                mask_size=28,
                pos_weight=-1,
                debug=False,
                mask_thr_binary=0.45,
            ),
            dict(
                assigner=dict(
                    type="MaxIoUAssigner",
                    pos_iou_thr=0.7,
                    neg_iou_thr=0.7,
                    min_pos_iou=0.7,
                    match_low_quality=False,
                    ignore_iof_thr=-1,
                ),
                sampler=dict(
                    type="RandomSampler",
                    num=512,
                    pos_fraction=0.25,
                    neg_pos_ub=-1,
                    add_gt_as_proposals=True,
                ),
                mask_size=28,
                pos_weight=-1,
                debug=False,
                mask_thr_binary=0.45,
            ),
        ],
    ),
    test_cfg=dict(
        rpn=dict(
            nms_pre=2000,
            max_per_img=1000,
            nms=dict(type="nms", iou_threshold=bbox_iou_threshold),
            min_bbox_size=0,
        ),
        rcnn=dict(
            score_thr=0.25,
            nms=dict(type="nms", iou_threshold=mask_iou_threshold),
            max_per_img=1000,
            mask_thr_binary=-1,
        ),
    ),
)
