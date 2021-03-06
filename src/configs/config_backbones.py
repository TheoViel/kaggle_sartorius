out_channels = {
    "swin_tiny": [96, 192, 384, 768],
    "swin_small": [96, 192, 384, 768],
    "swin_base": [128, 256, 512, 1024],
    "efficientnetv2_s": [48, 64, 160, 256],
    "efficientnetv2_m": [48, 80, 176, 512],
    "efficientnetv2_l": [64, 96, 224, 640],
    "efficientnet_b3": [32, 48, 136, 384],
    "efficientnet_b4": [32, 56, 160, 448],
    "efficientnet_b5": [40, 64, 176, 512],
    "efficientnet_b6": [40, 72, 200, 576],
    "efficientnet_b7": [48, 80, 224, 640],
}

custom_imports = dict(
    imports=['model_zoo.efficientnet'],
    allow_failed_imports=True
)

backbones = dict(
    resnet50=dict(
        type="ResNet",
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=-1,  # 1
        norm_cfg=dict(type="BN", requires_grad=True),
        norm_eval=True,
        style="pytorch",
    ),
    resnet101=dict(
        type="ResNet",
        depth=101,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=-1,  # 1
        norm_cfg=dict(type="BN", requires_grad=True),
        norm_eval=True,
        style="pytorch",
    ),
    resnext101=dict(
        type="ResNeXt",
        depth=101,
        groups=32,
        base_width=4,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=-1,  # 1
        norm_cfg=dict(type="BN", requires_grad=True),
        style="pytorch",
    ),
    resnext101_64x4=dict(
        type='ResNeXt',
        depth=101,
        groups=64,
        base_width=4,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=-1,
        norm_cfg=dict(type='BN', requires_grad=True),
        style='pytorch'
    ),
    resnext101_32x8=dict(
        type='ResNeXt',
        depth=101,
        groups=32,
        base_width=8,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        style='pytorch'
    ),
    swin_tiny=dict(
        type='SwinTransformer',
        patch_size=2,  # 4
        strides=(2, 2, 2, 2),
        embed_dims=96,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        window_size=7,
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.2,
        patch_norm=True,
        out_indices=(0, 1, 2, 3),
        with_cp=False,
        convert_weights=True,
    ),
    swin_small=dict(
        type='SwinTransformer',
        patch_size=4,  # 2
        strides=(4, 2, 2, 2),
        embed_dims=96,
        depths=[2, 2, 18, 2],
        num_heads=[3, 6, 12, 24],
        window_size=7,
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.3,
        patch_norm=True,
        out_indices=(0, 1, 2, 3),
        with_cp=False,
        convert_weights=True,
    ),
    swin_base=dict(
        type='SwinTransformer',
        patch_size=2,  # 4
        strides=(2, 2, 2, 2),
        embed_dims=128,
        depths=[2, 2, 18, 2],
        num_heads=[4, 8, 16, 32],
        window_size=7,
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.5,
        patch_norm=True,
        out_indices=(0, 1, 2, 3),
        with_cp=False,
        convert_weights=True,
    ),
    resnext50_gnws=dict(
        type='ResNeXt',
        depth=50,
        groups=32,
        base_width=4,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=-1,
        style='pytorch',
        conv_cfg=dict(type='ConvWS'),
        norm_cfg=dict(type='GN', num_groups=32, requires_grad=True),
        # init_cfg=dict(
        #     type='Pretrained',
        #     checkpoint='open-mmlab://jhu/resnext50_32x4d_gn_ws'
        # )
    ),
    resnext50=dict(
        type='ResNeXt',
        depth=50,
        groups=32,
        base_width=4,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=-1,
        style='pytorch',
        norm_cfg=dict(type='BN', requires_grad=True),
        init_cfg=dict(
            type='Pretrained',
            checkpoint='open-mmlab://resnext50_32x4d'
        )
    ),
    efficientnetv2_s=dict(
        type="EfficientNet",
        name='tf_efficientnetv2_s_in21ft1k',
        blocks_idx=[1, 2, 4, 5],
        pretrained=True,
    ),
    efficientnetv2_m=dict(
        type="EfficientNet",
        name='tf_efficientnetv2_m_in21ft1k',
        blocks_idx=[1, 2, 4, 6],
        pretrained=True,
    ),
    efficientnetv2_l=dict(
        type="EfficientNet",
        name='tf_efficientnetv2_l_in21ft1k',
        blocks_idx=[1, 2, 4, 6],
        pretrained=True,
    ),
    efficientnet_b3=dict(
        type="EfficientNet",
        name='tf_efficientnet_b3_ns',
        blocks_idx=[1, 2, 4, 6],
        pretrained=True,
    ),
    efficientnet_b4=dict(
        type="EfficientNet",
        name='tf_efficientnet_b4_ns',
        blocks_idx=[1, 2, 4, 6],
        pretrained=True,
    ),
    efficientnet_b5=dict(
        type="EfficientNet",
        name='tf_efficientnet_b5_ns',
        blocks_idx=[1, 2, 4, 6],
        pretrained=True,
    ),
    efficientnet_b6=dict(
        type="EfficientNet",
        name='tf_efficientnet_b6_ns',
        blocks_idx=[1, 2, 4, 6],
        pretrained=True,
    ),
    efficientnet_b7=dict(
        type="EfficientNet",
        name='tf_efficientnet_b7_ns',
        blocks_idx=[1, 2, 4, 6],
        pretrained=True,
    ),
)
