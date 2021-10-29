from mmcv import Config
from mmcv.utils import build_from_cfg
from mmdet.datasets.builder import PIPELINES
from mmdet.datasets.pipelines import Compose


def define_pipelines(config_file):
    pipe_cfg = Config.fromfile(config_file).data
    pipelines = {
        k: Compose(
            [build_from_cfg(aug, PIPELINES, None) for aug in pipe_cfg[k].pipeline]
        ) for k in pipe_cfg
    }
    return pipelines
