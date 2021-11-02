import sys
import logging

from mmcv import Config
from mmcv.parallel import MMDataParallel
from mmdet.models.builder import build_detector

from utils.torch import load_model_weights  # noqa  TODO : pretrain


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
        # TODO : do not hardcode path

    return model
