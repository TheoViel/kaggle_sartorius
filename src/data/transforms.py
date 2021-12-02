from mmcv import Config
from mmcv.utils import build_from_cfg
from mmdet.datasets.builder import PIPELINES
from mmdet.datasets.pipelines import Compose
from mmdet.datasets.dataset_wrappers import MultiImageMixDataset


def define_pipelines(config_file, multi_image=False):
    """
    Defines the MMDet augmentation pipelines from a config file.

    Args:
        config_file (str): Path to the config file.
        multi_image (bool, optional): Whether to use a multi image dataset. Defaults to False.

    Returns:
        dict of MMDet pipelines: Pipelines for train, val & test.
    """
    pipe_cfg = Config.fromfile(config_file).data

    if not multi_image:
        pipelines = {
            k: Compose(
                [build_from_cfg(aug, PIPELINES, None) for aug in pipe_cfg[k].pipeline]
            ) for k in pipe_cfg
        }
    else:
        pipelines = {
            k: pipe_cfg[k].pipeline for k in pipe_cfg
        }
    return pipelines


def to_mosaic(config, dataset, pipeline_name='mosaic'):
    """
    TODO

    Args:
        config ([type]): [description]
        dataset ([type]): [description]
        pipeline_name (str, optional): [description]. Defaults to 'mosaic'.

    Returns:
        [type]: [description]
    """
    pipelines_dicts = define_pipelines(config.data_config, multi_image=True)

    dataset.CLASSES = []
    dataset_mosaic = MultiImageMixDataset(dataset, pipelines_dicts[pipeline_name])

    return dataset_mosaic
