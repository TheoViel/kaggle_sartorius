import ast
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold

from params import DATA_PATH, OUT_PATH, TRAIN_IMG_PATH
from params import LIVECELL_PATH, LIVECELL_FOLDERS, LIVECELL_CLASSES

WRONG_ANNOTATIONS = ["03b27b381a5f", "e92c56871769", "eec79772cb99"]


def prepare_data(fix=False, remove_anomalies=False):
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

    return df


def prepare_extra_data(name):
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


def get_splits(df, config):
    if config.split == "skf":
        skf = StratifiedKFold(
            n_splits=config.k, shuffle=True, random_state=config.random_state
        )
        splits = list(skf.split(X=df, y=df["cell_type"]))
    elif config.split == "sgkf":
        sgkf = StratifiedGroupKFold(
            n_splits=config.k, shuffle=True, random_state=config.random_state
        )
        splits = list(sgkf.split(X=df, y=df["cell_type"], groups=df["sample_id"]))

    return splits
