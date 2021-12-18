import os
import ast
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, GroupKFold

from params import DATA_PATH, OUT_PATH, TRAIN_IMG_PATH


WRONG_ANNOTATIONS = ["03b27b381a5f", "e92c56871769", "eec79772cb99"]


def plate_to_class(plate):
    mapping = {
        "astro[hippo]": 0,
        "astros[cereb]": 1,
        "cort[6-OHDA]": 2,
        "cort[debris]": 4,
        "cort[density]": 3,
        "cort[oka-high]": 4,
        "cort[oka-low]": 2,
        "cort[pre-treat]": 3,
        "shsy5y[diff]": 5,
    }
    return mapping[plate]


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

    df['plate_class'] = df['plate'].apply(plate_to_class)

    df['is_extra'] = 0

    return df


def prepare_extra_data():
    """
    Prepares the pseudo labeled data.

    Args:
        name (str): Name of the csv file.

    Returns:
        pandas DataFrame: metadata.
    """

    df_extra = pd.DataFrame({'id': os.listdir(DATA_PATH + "train_semi_supervised/")})
    df_extra['predicted'] = ""
    df_extra['img_path'] = DATA_PATH + "train_semi_supervised/" + df_extra['id']

    df_extra['plate'] = df_extra['id'].apply(lambda x: x.split('_')[0])
    df_extra['well'] = df_extra['id'].apply(lambda x: x.split('_')[1][:-2])
    df_extra['plate_well'] = df_extra['plate'] + "_" + df_extra['well']

    df_extra['cell_type'] = df_extra['id'].apply(lambda x: x.split('[')[0])
    df_extra['cell_type'] = df_extra['cell_type'].apply(lambda x: "astro" if x == "astros" else x)

    df_extra['plate_class'] = df_extra['plate'].apply(plate_to_class)

    df_extra['is_extra'] = 1

    return df_extra


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
        splits = list(skf.split(X=df, y=df["plate"]))
    elif config.split == "sgkf":
        raise NotImplementedError
        # sgkf = StratifiedGroupKFold(
        #     n_splits=config.k, shuffle=True, random_state=config.random_state
        # )
        # splits = list(sgkf.split(X=df, y=df["cell_type"], groups=df["sample_id"]))
    elif config.split == "gkf":
        gkf = GroupKFold(n_splits=config.k)
        splits = list(gkf.split(X=df, groups=df["plate_well"]))

    return splits
