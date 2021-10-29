import sys
import logging

from mmcv import Config
from mmcv.parallel import MMDataParallel
from mmdet.models.builder import build_detector

from utils.torch import load_model_weights  # noqa  TODO : pretrain


def define_model(config_file, reduce_stride=False, pretrained=False, verbose=0):
    model_cfg = Config.fromfile("model_zoo/config.py")

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
        # pass
        model.backbone.conv1.stride = (1, 1)
        # model.roi_head.mask_head.upsample.stride = (1, 1)
        # model.roi_head.mask_head.scale_factor = 1

        # for block in [model.roi_head.mask_roi_extractor, model.roi_head.bbox_roi_extractor]:
        #     for roi_layer in block.roi_layers:
        #         roi_layer.spatial_scale *= 2

        # change roialign ?

    model = MMDataParallel(model)

    if pretrained:
        model = load_model_weights(model, '../logs/2021-10-29/2/maskrcnn_0.pt', verbose=1)

    return model
