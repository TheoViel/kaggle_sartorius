import sys
import torch
import logging

from mmcv import Config
from mmcv.parallel import MMDataParallel
from mmdet.models.builder import build_detector

from utils.torch import load_model_weights


def define_model(config_file, reduce_stride=False, pretrained_weights=None, verbose=0):
    model_cfg = Config.fromfile(config_file)

    model = build_detector(
        model_cfg.model,
        model_cfg.get('train_cfg'),
        model_cfg.get('test_cfg')
    ).cpu()

    if not verbose:
        logging.disable(sys.maxsize)

    model.init_weights()

    if not verbose:  # re-enable
        logging.disable(logging.NOTSET)

    if reduce_stride:
        model.backbone.conv1.stride = (1, 1)

    model = MMDataParallel(model)

    if pretrained_weights is not None:
        model = load_model_weights(model, pretrained_weights, verbose=1)
    elif "pretrained_weights" in model_cfg.keys():
        print(f"\n -> Loading weights from {model_cfg['pretrained_weights']}\n")

        dic = torch.load(model_cfg['pretrained_weights'])['state_dict']

        del dic['roi_head.bbox_head.fc_cls.weight']
        del dic['roi_head.bbox_head.fc_cls.bias']
        del dic['roi_head.bbox_head.fc_reg.weight']
        del dic['roi_head.bbox_head.fc_reg.bias']
        del dic['roi_head.mask_head.conv_logits.weight']
        del dic['roi_head.mask_head.conv_logits.bias']

        model.module.load_state_dict(dic, strict=False)

    return model
