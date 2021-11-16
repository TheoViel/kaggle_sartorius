import sys
import mmcv
import torch
import logging

from mmcv.parallel import MMDataParallel
from mmdet.models.builder import build_detector


def define_model(config_file, encoder="resnet50", pretrained_livecell=False, verbose=1):
    cfg = mmcv.Config.fromfile(config_file)

    cfg.model.backbone = cfg.backbones[encoder]

    model = build_detector(cfg.model)

    model.test_cfg = cfg["model"]["test_cfg"]
    model.train_cfg = cfg["model"]["train_cfg"]

    if "resnet" in encoder or "resnext" in encoder:  # reduce stride
        model.backbone.conv1.stride = (1, 1)

    model = MMDataParallel(model)

    # Weights
    weights = (
        cfg.pretrained_weights_livecell[encoder]
        if pretrained_livecell
        else cfg.pretrained_weights[encoder]
    )

    model = load_pretrained_weights(model, weights, verbose=verbose)

    return model


def load_pretrained_weights(model, weights, verbose=0):
    if verbose < 2:
        logging.disable(sys.maxsize)

    model.module.init_weights()

    if verbose < 2:  # re-enable
        logging.disable(logging.NOTSET)

    if verbose:
        print(f"\n -> Loading weights from {weights}\n")

    dic = torch.load(weights)
    if "state_dict" in dic.keys():  # coco pretrained
        dic = dic["state_dict"]

    for k in list(dic.keys()):
        if "fc_cls" in k or "fc_reg" in k or "conv_logits" in k:
            del dic[k]

    if "module" in list(dic.keys())[0]:  # pretrained_livecell=True
        incompatible_keys = model.load_state_dict(dic, strict=False)
    else:
        incompatible_keys = model.module.load_state_dict(dic, strict=False)

    assert len(incompatible_keys.unexpected_keys) == 0, "Unexpected keys in dict"
    for k in incompatible_keys.missing_keys:
        assert "fc_cls" in k or "fc_reg" in k or "conv_logits" in k, f"Missing key in dict: {k}"

    return model
