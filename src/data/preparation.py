import ast
import cv2
import pycocotools
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold, GroupKFold

from utils.rle import rle_decode, rles_to_mask_fix
from params import (
    DATA_PATH,
    OUT_PATH,
    TRAIN_IMG_PATH,
    ORIG_SIZE,
    LIVECELL_PATH,
    LIVECELL_FOLDERS,
    LIVECELL_CLASSES,
    CELL_TYPES,
    HCK_FIX_PATH
)

WRONG_ANNOTATIONS = ["03b27b381a5f", "e92c56871769", "eec79772cb99"]


def prepare_data(fix=False, remove_anomalies=False):
    """
    Prepares the data for training and validation.

    Args:
        fix (bool, optional): Whether to used fixed masks. Defaults to False.
        remove_anomalies (bool, optional): Whether to remove abnormal masks. Defaults to False.

    Returns:
        pandas DataFrame: metadata.
    """
    df = pd.read_csv(DATA_PATH + "train.csv")
    df = df.groupby('id').agg(list).reset_index()
    for col in df.columns[2:]:
        df[col] = df[col].apply(
            lambda x: np.unique(x)[0] if len(np.unique(x)) == 1 else np.unique(x)
        )
    df['img_path'] = TRAIN_IMG_PATH + df['id'] + ".png"

    if fix:
        df_mmdet = pd.read_csv(OUT_PATH + "mmdet_data_fix.csv")  # mmdet_data.csv
    else:
        df_mmdet = pd.read_csv(OUT_PATH + "mmdet_data_nofix.csv")

    df_mmdet['ann'] = df_mmdet['ann'].apply(ast.literal_eval)
    df_mmdet['id'] = df_mmdet['filename'].apply(lambda x: x[:-4])

    df = df.merge(df_mmdet[['id', 'ann']], on='id')

    df['is_extra'] = 0

    if remove_anomalies:
        df = df.loc[~df['id'].isin(WRONG_ANNOTATIONS)].reset_index(drop=True)

    df['plate'] = df['sample_id'].apply(lambda x: x.split('_')[0])
    df['well'] = df['sample_id'].apply(lambda x: x.split('_')[1][:-2])
    df['plate_well'] = df['plate'] + "_" + df['well']

    return df


def prepare_extra_data(name):
    """
    Prepares the livecell data.

    Args:
        name (str): Name of the csv file.

    Returns:
        pandas DataFrame: metadata.
    """
    df = pd.read_csv(OUT_PATH + name + ".csv")

    df['split'] = df['split'].apply(
        lambda x: "livecell_test_images/" if x == "test" else "livecell_train_val_images/"
    )
    df['cell_type'] = df['cell_type'].apply(lambda x: x.lower())
    df['cell_folder'] = df['cell_type'].apply(lambda x: LIVECELL_FOLDERS[LIVECELL_CLASSES.index(x)])

    df['img_path'] = LIVECELL_PATH + df['split'] + df['cell_folder'] + "/" + df['filename']

    df['ann'] = df['ann'].apply(ast.literal_eval)

    df['is_extra'] = 1

    return df


def prepare_pl_data(name):
    """
    Prepares the pseudo labeled data.

    Args:
        name (str): Name of the csv file.

    Returns:
        pandas DataFrame: metadata.
    """
    df = pd.read_csv(OUT_PATH + name + ".csv")

    df['cell_type'] = df['cell_type'].apply(lambda x: x.lower())

    df['img_path'] = DATA_PATH + "train_semi_supervised/" + df['filename']

    df['ann'] = df['ann'].apply(ast.literal_eval)

    df['is_extra'] = 1

    return df


def get_splits(df, config):
    """
    Computes the splits.
    Supports StratifiedKFold and StratifiedGroupKFold.
    Stratification is made per cell type, groups are made per sample id.

    Args:
        df (pandas DataFrame): Data to split.
        config (Config): Config that contains the split parameters.

    Returns:
        list: Splits as returned by the sklearn functions.
    """
    if config.split == "skf":
        skf = StratifiedKFold(
            n_splits=config.k, shuffle=True, random_state=config.random_state
        )
        splits = list(skf.split(X=df, y=df["cell_type"]))
    elif config.split == "sgkf":
        sgkf = StratifiedGroupKFold(
            n_splits=config.k, shuffle=True, random_state=config.random_state
        )
        splits = list(sgkf.split(X=df, y=df["cell_type"], groups=df["plate_well"]))
    elif config.split == "gkf":
        gkf = GroupKFold(n_splits=config.k)
        splits = list(gkf.split(X=df, groups=df["plate_well"]))

    return splits


def prepare_mmdet_data(df, idx, fix=True):
    """
    Processes the training data for mmdet.

    Args:
        df (pandas DataFrame): Train data.
        idx (int): Row to process.
        fix (bool, optional): Whether to fix masks. Defaults to True.

    Returns:
        np array: Masks.
        dict: Prepared data.
    """
    height, width = df[["height", "width"]].values[idx]
    cell_type = df['cell_type'][idx]

    rles = df['annotation'][idx]

    masks = rles_to_mask_fix(rles, (height, width), single_channel=False, fix=fix)

    if fix:
        mask_fix = cv2.imread(HCK_FIX_PATH + df['id'][idx] + ".png")
        if mask_fix is not None:
            masks = masks * (mask_fix[:, :, 2] > 0)[None]

    rles = [pycocotools.mask.encode(np.asfortranarray(m > 0)) for m in masks]

    bboxes = np.array([pycocotools.mask.toBbox(rle) for rle in rles])
    bboxes[:, 2] += bboxes[:, 0]
    bboxes[:, 3] += bboxes[:, 1]

    meta = {
        'filename': df['id'][idx] + ".png",
        'width': int(width),
        'height': int(height),
        'cell_type': cell_type,
        'ann': {
            'bboxes': bboxes.astype(int).tolist(),
            'labels': [CELL_TYPES.index(cell_type)] * len(bboxes),
            'masks': rles
        }
    }
    return masks, meta


def sub_to_mmdet(df, idx):
    """
    Adapts data in the submission format to the mmdet format.
    Used for pseudo labels.

    Args:
        df (pandas DataFrame): Data.
        idx (int): Row to process.

    Returns:
        np array: Masks.
        dict: Prepared data.
    """
    height, width = ORIG_SIZE
    cell_type = df['id'][idx].split('[')[0]

    if cell_type == "astros":
        cell_type = "astro"

    cell_type_idx = CELL_TYPES.index(cell_type) if cell_type in CELL_TYPES else -1

    rles = df['predicted'][idx]
    masks = np.array([rle_decode(enc, ORIG_SIZE) for enc in rles])
    rles = [pycocotools.mask.encode(np.asfortranarray(m > 0)) for m in masks]

    bboxes = np.array([pycocotools.mask.toBbox(rle) for rle in rles])
    bboxes[:, 2] += bboxes[:, 0]
    bboxes[:, 3] += bboxes[:, 1]

    meta = {
        'filename': df['id'][idx],
        'width': int(width),
        'height': int(height),
        'cell_type': cell_type,
        'ann': {
            'bboxes': bboxes.astype(int).tolist(),
            'labels': [cell_type_idx] * len(bboxes),
            'masks': rles
        }
    }
    return masks, meta