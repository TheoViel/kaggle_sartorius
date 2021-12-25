import numpy as np
import pandas as pd

from params import DATA_PATH, TRAIN_IMG_PATH


WRONG_ANNOTATIONS = ["03b27b381a5f", "e92c56871769", "eec79772cb99"]


def prepare_data(remove_anomalies=True):
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

    if remove_anomalies:
        df = df.loc[~df['id'].isin(WRONG_ANNOTATIONS)].reset_index(drop=True)

    df['plate'] = df['sample_id'].apply(lambda x: x.split('_')[0])
    df['well'] = df['sample_id'].apply(lambda x: x.split('_')[1][:-2])
    df['plate_well'] = df['plate'] + "_" + df['well']

    return df


def get_plate_wells(paths):
    df = prepare_data()
    df = df[['id', 'plate_well']].set_index('id')

    ids = [p.split('/')[-1].split('_')[1] for p in paths]
    plate_wells = [df.loc[id_]['plate_well'] for id_ in ids]

    return plate_wells
