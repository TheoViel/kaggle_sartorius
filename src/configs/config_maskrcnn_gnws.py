# https://github.com/open-mmlab/mmdetection/blob/master/configs/_base_/models/mask_rcnn_r50_fpn.py

num_classes = 8  # 3 + 7 for training, 8 for pretraining
mask_iou_threshold = 0.3
bbox_iou_threshold = 0.7
roi_mask_size = 14

pretrained_weights = {
    "resnext50_gnws": "../input/weights/mask_rcnn_x50_32x4d_fpn_gn_ws.pth",
    "resnext101_gnws": "../input/weights/mask_rcnn_x101_32x4d_fpn_gn_ws.pth",
}

pretrained_weights_livecell = {
    "resnext50_gnws": "../logs/pretrain/2021-12-27/0/maskrcnn_gnws_resnext50_gnws_0.pt",
    "resnext101_gnws": None,
}

conv_cfg = dict(type="ConvWS")
norm_cfg = dict(type="GN", num_groups=32, requires_grad=True)

model = dict(
    type="MaskRCNN",
    backbone="",
    neck=dict(
        type="FPN",
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        num_outs=5,
        conv_cfg=conv_cfg,
        norm_cfg=norm_cfg,
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
        loss_bbox=dict(type="L1Loss", loss_weight=1.0),
    ),
    roi_head=dict(
        type="StandardRoIHead",
        bbox_roi_extractor=dict(
            type="SingleRoIExtractor",
            roi_layer=dict(type="RoIAlign", output_size=7, sampling_ratio=0),
            out_channels=256,
            featmap_strides=[2, 4, 8, 16],
        ),
        bbox_head=dict(
            type="Shared4Conv1FCBBoxHead",
            in_channels=256,
            fc_out_channels=1024,
            roi_feat_size=7,
            conv_out_channels=256,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            num_classes=num_classes,
            bbox_coder=dict(
                type="DeltaXYWHBBoxCoder",
                target_means=[0.0, 0.0, 0.0, 0.0],
                target_stds=[0.1, 0.1, 0.2, 0.2],
            ),
            reg_class_agnostic=False,
            loss_cls=dict(type="CrossEntropyLoss", use_sigmoid=False, loss_weight=1.0),
            loss_bbox=dict(type="L1Loss", loss_weight=1.0),
        ),
        mask_roi_extractor=dict(
            type="SingleRoIExtractor",
            roi_layer=dict(
                type="RoIAlign", output_size=roi_mask_size, sampling_ratio=0
            ),
            out_channels=256,
            featmap_strides=[2, 4, 8, 16],
        ),
        mask_head=dict(
            type="FCNMaskHead",
            num_convs=4,
            in_channels=256,
            conv_out_channels=256,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            num_classes=num_classes,
            loss_mask=dict(type="CrossEntropyLoss", use_mask=True, loss_weight=1.0),
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
            max_per_img=1000,
            nms=dict(type="nms", iou_threshold=bbox_iou_threshold),
            min_bbox_size=0,
        ),
        rcnn=dict(
            assigner=dict(
                type="MaxIoUAssigner",
                pos_iou_thr=0.5,
                neg_iou_thr=0.5,
                min_pos_iou=0.5,
                match_low_quality=True,
                ignore_iof_thr=-1,
            ),
            sampler=dict(
                type="RandomSampler",
                num=512,
                pos_fraction=0.25,
                neg_pos_ub=-1,
                add_gt_as_proposals=True,
            ),
            mask_size=roi_mask_size * 2,
            pos_weight=-1,
            debug=False,
        ),
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
