import ast
import numpy as np
import pandas as pd

from params import DATA_PATH, OUT_PATH, TRAIN_IMG_PATH


def prepare_data(fix=False):
    df = pd.read_csv(DATA_PATH + "train.csv")
    df = df.groupby('id').agg(list).reset_index()
    for col in df.columns[2:]:
        df[col] = df[col].apply(
            lambda x: np.unique(x)[0] if len(np.unique(x)) == 1 else np.unique(x)
        )
    df['img_path'] = TRAIN_IMG_PATH + df['id'] + ".png"

    if fix:
        df_mmdet = pd.read_csv(OUT_PATH + "mmdet_data.csv")
    else:
        df_mmdet = pd.read_csv(OUT_PATH + "mmdet_data_nofix.csv")

    df_mmdet['ann'] = df_mmdet['ann'].apply(ast.literal_eval)
    df_mmdet['id'] = df_mmdet['filename'].apply(lambda x: x[:-4])

    df = df.merge(df_mmdet[['id', 'ann']], on='id')

    return df
