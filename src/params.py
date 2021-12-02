# Hardcoded stuff, paths are to adapt to your setup

import torch
import numpy as np

NUM_WORKERS = 0

DATA_PATH = "../input/"
TRAIN_IMG_PATH = DATA_PATH + "train/"
HCK_FIX_PATH = DATA_PATH + "hck_fix/"

LOG_PATH = "../logs/"
OUT_PATH = "../output/"

MEAN = np.array([0.66437738, 0.50478148, 0.70114894])
STD = np.array([0.15825711, 0.24371008, 0.13832686])

CELL_TYPES = ["shsy5y", "astro", "cort"]
NUM_CLASSES = len(CELL_TYPES)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

ORIG_SIZE = (520, 704)

LIVECELL_PATH = DATA_PATH + "LIVECell_dataset_2021/images/"
LIVECELL_CLASSES = ['skbr3', 'a172', 'huh7', 'bv2', 'mcf7', 'bt474', 'skov3', 'shsy5y']
LIVECELL_FOLDERS = ['SkBr3', 'A172', 'Huh7', 'BV2', 'MCF7', 'BT474', 'SKOV3', 'SHSY5Y']

MAX_CELLS_PER_IMG = [800, 600, 150]
