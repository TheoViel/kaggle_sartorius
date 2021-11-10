import sys
import torch
import logging

from mmcv import Config
from mmcv.parallel import MMDataParallel
from mmdet.models.builder import build_detector

from utils.torch import load_model_weights


def define_model(config_file, encoder="resnet50", pretrained_weights=None, verbose=0):
    cfg = Config.fromfile(config_file)

    cfg.model.backbone = cfg.backbones[encoder]

    try:
        weights = cfg.pretrained_weights[encoder]
    except KeyError:
        weights = None

    model = build_detector(cfg.model)

    model.test_cfg = cfg['model']['test_cfg']
    model.train_cfg = cfg['model']['train_cfg']

    if not verbose:
        logging.disable(sys.maxsize)

    model.init_weights()

    if not verbose:  # re-enable
        logging.disable(logging.NOTSET)

    if "resnet" in encoder or 'resnext' in encoder:
        model.backbone.conv1.stride = (1, 1)

    model = MMDataParallel(model)

    if pretrained_weights is not None:
        model = load_model_weights(model, pretrained_weights, verbose=1)
    elif weights is not None:
        print(f"\n -> Loading weights from {weights}\n")

        dic = torch.load(weights)['state_dict']

        del dic['roi_head.bbox_head.fc_cls.weight']
        del dic['roi_head.bbox_head.fc_cls.bias']
        del dic['roi_head.bbox_head.fc_reg.weight']
        del dic['roi_head.bbox_head.fc_reg.bias']
        del dic['roi_head.mask_head.conv_logits.weight']
        del dic['roi_head.mask_head.conv_logits.bias']

        model.module.load_state_dict(dic, strict=False)

    return model
