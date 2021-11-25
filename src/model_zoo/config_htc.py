# https://github.com/open-mmlab/mmdetection/blob/master/configs/htc/htc_r50_fpn_1x_coco.py

num_classes = 3  # 3 for training, 8 for pretraining
mask_iou_threshold = 0.3
bbox_iou_threshold = 0.7

pretrained_weights = {
    "resnet50": "../input/weights/htc_r50_fpn_20e_coco.pth",
    "resnext101": "../input/weights/htc_x101_32x4d_fpn_16x1_20e_coco.pth",
}

pretrained_weights_livecell = {
    "resnet50": "../logs/pretrain/2021-11-24/1/htc_resnet50_0.pt",
    "resnext101": "../logs/pretrain/2021-11-24/2/htc_resnext101_0.pt",
}


model = dict(
    type="HybridTaskCascade",
    backbone="",
    neck=dict(type="FPN", in_channels=[256, 512, 1024, 2048], out_channels=256, num_outs=5),
    rpn_head=dict(
        type="RPNHead",
        in_channels=256,
        feat_channels=256,
        anchor_generator=dict(
            type="AnchorGenerator", scales=[8], ratios=[0.5, 1.0, 2.0], strides=[2, 4, 8, 16, 32]
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
        type="HybridTaskCascadeRoIHead",
        interleaved=True,
        mask_info_flow=True,
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
                loss_cls=dict(type="CrossEntropyLoss", use_sigmoid=False, loss_weight=1.0),
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
                loss_cls=dict(type="CrossEntropyLoss", use_sigmoid=False, loss_weight=1.0),
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
                loss_cls=dict(type="CrossEntropyLoss", use_sigmoid=False, loss_weight=1.0),
                loss_bbox=dict(type="SmoothL1Loss", beta=1.0, loss_weight=1.0),
            ),
        ],
        mask_roi_extractor=dict(
            type="SingleRoIExtractor",
            roi_layer=dict(type="RoIAlign", output_size=14, sampling_ratio=0),
            out_channels=256,
            featmap_strides=[2, 4, 8, 16],
        ),
        mask_head=[
            dict(
                type="HTCMaskHead",
                with_conv_res=False,
                num_convs=4,
                in_channels=256,
                conv_out_channels=256,
                num_classes=num_classes,
                loss_mask=dict(type="CrossEntropyLoss", use_mask=True, loss_weight=1.0),
            ),
            dict(
                type="HTCMaskHead",
                num_convs=4,
                in_channels=256,
                conv_out_channels=256,
                num_classes=num_classes,
                loss_mask=dict(type="CrossEntropyLoss", use_mask=True, loss_weight=1.0),
            ),
            dict(
                type="HTCMaskHead",
                num_convs=4,
                in_channels=256,
                conv_out_channels=256,
                num_classes=num_classes,
                loss_mask=dict(type="CrossEntropyLoss", use_mask=True, loss_weight=1.0),
            ),
        ],
        semantic_roi_extractor=dict(
            type="SingleRoIExtractor",
            roi_layer=dict(type="RoIAlign", output_size=14, sampling_ratio=0),
            out_channels=256,
            featmap_strides=[4],
        ),
        semantic_head=dict(
            type="FusedSemanticHead",
            num_ins=5,
            fusion_level=1,
            num_convs=4,
            in_channels=256,
            conv_out_channels=256,
            num_classes=num_classes,  # 183 ??
            loss_seg=dict(type="CrossEntropyLoss", ignore_index=255, loss_weight=0.2),
        ),
    ),
    # model training and testing settings
    train_cfg=dict(
        rpn=dict(
            assigner=dict(
                type="MaxIoUAssigner",
                pos_iou_thr=0.7,
                neg_iou_thr=0.3,
                min_pos_iou=0.3,
                ignore_iof_thr=-1,
            ),
            sampler=dict(
                type="RandomSampler",
                num=256,
                pos_fraction=0.5,
                neg_pos_ub=-1,
                add_gt_as_proposals=False,
            ),
            allowed_border=0,
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
            ),
            dict(
                assigner=dict(
                    type="MaxIoUAssigner",
                    pos_iou_thr=0.6,
                    neg_iou_thr=0.6,
                    min_pos_iou=0.6,
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
            ),
            dict(
                assigner=dict(
                    type="MaxIoUAssigner",
                    pos_iou_thr=0.7,
                    neg_iou_thr=0.7,
                    min_pos_iou=0.7,
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
