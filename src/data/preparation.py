import numpy as np
import pandas as pd

from params import DATA_PATH, OUT_PATH, TRAIN_IMG_PATH


def prepare_data(width=1):
    df = pd.read_csv(DATA_PATH + "train.csv")
    df = df.groupby('id').agg(list).reset_index()
    for col in df.columns[2:]:
        df[col] = df[col].apply(
            lambda x: np.unique(x)[0] if len(np.unique(x)) == 1 else np.unique(x)
        )
    df['img_path'] = TRAIN_IMG_PATH + df['id'] + ".png"
    df['mask_path'] = OUT_PATH + f"proc_train_{width}/" + df['id'] + ".npy"

    return df
