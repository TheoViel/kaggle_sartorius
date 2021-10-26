import sys
import logging

from mmcv import Config
from mmcv.parallel import MMDataParallel
from mmdet.models.builder import build_detector


def define_model(config_file, verbose=0):
    model_cfg = Config.fromfile("model_zoo/config.py")

    model = build_detector(model_cfg.model).cpu()

    if not verbose:
        logging.disable(sys.maxsize)

    model.init_weights()

    if not verbose:  # re-enable
        logging.disable(logging.NOTSET)

    model = MMDataParallel(model)

    return model
