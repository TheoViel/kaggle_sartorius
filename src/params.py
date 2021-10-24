# Hardcoded stuff, paths are to adapt to your setup

import torch
import numpy as np

NUM_WORKERS = 4

DATA_PATH = "../input/"
TRAIN_IMG_PATH = DATA_PATH + "train/"

LOG_PATH = "../logs/"
OUT_PATH = "../output/"

CLASSES = ["cell"]
NUM_CLASSES = len(CLASSES)

MEAN = np.array([0.66437738, 0.50478148, 0.70114894])
STD = np.array([0.15825711, 0.24371008, 0.13832686])

CELL_TYPES = ['shsy5y', 'astro', 'cort']

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

SIZE = 256

ORIG_SIZE = (520, 704)
