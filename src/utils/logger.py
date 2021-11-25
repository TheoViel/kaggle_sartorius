import os
import sys
import json
import shutil
import datetime
import numpy as np


class Config:
    """
    Placeholder to load a config from a saved json
    """
    def __init__(self, dic):
        for k, v in dic.items():
            setattr(self, k, v)


class Logger(object):
    """
    Simple logger that saves what is printed in a file
    """

    def __init__(self, *files):
        self.files = files

    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush()

    def flush(self):
        for f in self.files:
            f.flush()


def create_logger(directory="", name="logs.txt"):
    """
    Creates a logger to log output in a chosen file

    Args:
        directory (str, optional): Path to save logs at. Defaults to "".
        name (str, optional): Name of the file to save the logs in. Defaults to "logs.txt".
    """

    log = open(directory + name, "a", encoding="utf-8")
    file_logger = Logger(sys.stdout, log)

    sys.stdout = file_logger
    sys.stderr = file_logger


def prepare_log_folder(log_path):
    """
    Creates the directory for logging.
    Logs will be saved at log_path/date_of_day/exp_id

    Args:
        log_path (str): Directory

    Returns:
        str: Path to the created log folder
    """
    today = str(datetime.date.today())
    log_today = f"{log_path}{today}/"

    if not os.path.exists(log_today):
        os.mkdir(log_today)

    exp_id = (
        np.max([int(f) for f in os.listdir(log_today)]) + 1
        if len(os.listdir(log_today))
        else 0
    )
    log_folder = log_today + f"{exp_id}/"

    assert not os.path.exists(log_folder), "Experiment already exists"
    os.mkdir(log_folder)

    return log_folder


def save_config(config, folder):
    """
    Saves a config as a json, copies data and model configs.

    Args:
        config (Config): Config.
        path (str): Path to save at.

    Returns:
        pandas dataframe: Config as a dataframe
    """
    data_config_file = folder + config.data_config.split('/')[-1]
    model_config_file = folder + config.model_config.split('/')[-1]

    backbone_config_file_old = config.model_config.rsplit('/', 1)[0] + "/config_backbones.py"
    backbone_config_file = folder + "config_backbones.py"

    shutil.copyfile(config.data_config, data_config_file)
    shutil.copyfile(config.model_config, model_config_file)
    shutil.copyfile(backbone_config_file_old, backbone_config_file)

    config.data_config = data_config_file
    config.model_config_file = model_config_file

    dic = config.__dict__.copy()
    del dic["__doc__"], dic["__module__"], dic["__dict__"], dic["__weakref__"]

    with open(folder + "config.json", "w") as f:
        json.dump(dic, f)
