import re
import sys
import mmcv
import torch
import logging

from collections import OrderedDict
from mmcv.parallel import MMDataParallel
from mmdet.models.builder import build_detector


def define_model(config_file, encoder="resnet50", pretrained_livecell=False, verbose=1):
    # Configs
    cfg = mmcv.Config.fromfile(config_file)

    config_backbone_file = config_file.rsplit('/', 1)[0] + "/config_backbones.py"
    cfg_backbones = mmcv.Config.fromfile(config_backbone_file)
    cfg.model.backbone = cfg_backbones.backbones[encoder]

    if encoder in cfg_backbones.out_channels.keys():  # update neck channels
        cfg.model.neck.in_channels = cfg_backbones.out_channels[encoder]

    # Build model
    model = build_detector(cfg.model)
    model.test_cfg = cfg["model"]["test_cfg"]
    model.train_cfg = cfg["model"]["train_cfg"]

    # Reduce stride
    if "resnet" in encoder or "resnext" in encoder:
        model.backbone.conv1.stride = (1, 1)
    elif "efficientnet" in encoder:
        model.backbone.effnet.conv_stem.stride = (1, 1)

    model = MMDataParallel(model)

    # Weights
    try:
        weights = (
            cfg.pretrained_weights_livecell[encoder]
            if pretrained_livecell
            else cfg.pretrained_weights[encoder]
        )
    except KeyError:
        weights = None

    model = load_pretrained_weights(
        model,
        weights,
        verbose=verbose,
        adapt_swin="swin" in encoder and not pretrained_livecell
    )

    return model


def load_pretrained_weights(model, weights, verbose=0, adapt_swin=False):
    if verbose < 2:
        logging.disable(sys.maxsize)

    model.module.init_weights()

    if verbose < 2:  # re-enable
        logging.disable(logging.NOTSET)

    if weights is None:
        return model

    if verbose:
        print(f"\n -> Loading weights from {weights}\n")

    dic = torch.load(weights)
    if "state_dict" in dic.keys():  # coco pretrained
        dic = dic["state_dict"]

    # Swin
    if adapt_swin:
        dic = adapt_swin_weights(dic)

    # Remove classification layers
    for k in list(dic.keys()):
        if "fc_cls" in k or "fc_reg" in k or "conv_logits" in k:
            del dic[k]

    # Handle MMDataParallel wrapper
    if "module" in list(dic.keys())[0]:  # pretrained_livecell=True
        incompatible_keys = model.load_state_dict(dic, strict=False)
    else:
        incompatible_keys = model.module.load_state_dict(dic, strict=False)

    allowed_missing = ["fc_cls", "fc_reg", "conv_logits", "mask_iou_head"]
    if adapt_swin:
        allowed_missing += ["roi_head"]

    for k in incompatible_keys.missing_keys:
        assert any([allowed in k for allowed in allowed_missing]), f"Missing key in dict: {k}"

    return model


def adapt_swin_weights(w):
    new_w = OrderedDict()

    for k in w:
        if "backbone" not in k:
            if "roi_head.bbox_head" not in k:  # ignore roi_head
                new_w[k] = w[k]
        else:
            # rename
            new_k = re.sub("layers", "stages", k)
            new_k = re.sub("patch_embed.proj", "patch_embed.projection", new_k)
            new_k = re.sub("attn.", "attn.w_msa.", new_k)
            new_k = re.sub("mlp.", "ffn.", new_k)
            new_k = re.sub("ffn.fc1", "ffn.layers.0.0", new_k)
            new_k = re.sub("ffn.fc2", "ffn.layers.1", new_k)

            new_w[new_k] = w[k]

    # reduce stride
    new_w["backbone.patch_embed.projection.weight"] = new_w[
        "backbone.patch_embed.projection.weight"
    ][:, :, :2, :2]

    return new_w
