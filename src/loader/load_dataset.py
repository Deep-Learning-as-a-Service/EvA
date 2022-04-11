import itertools
import os

import numpy as np
import pandas as pd

from utils.Recording import Recording
import utils.settings as settings


def load_dataset(path: str, label_column_name: str, recording_idx_name: str, column_names_to_ignore: 'list[str]') -> "list[Recording]":
    """
    Returns a list of Recordings from the dataset
    """
    dataset = pd.read_csv(path, header=0, engine='python')

    assert len(df.columns) == len(set(df.columns)), "Duplicate column names"
    feature_column_names = list(dataset.columns) - [label_column_name, recording_idx_name] - column_names_to_ignore

    recording_idxs = np.array(dataset.loc[:, recording_idx_name])
    recording_change_idxs = np.where(recording_idxs[:-1] != recording_idxs[1:])[0] + 1

    start_idx = 0
    recordings = []
    for end_idx in recording_change_idxs:
        recordings.append(Recording(
            sensor_frame = dataset.loc[start:(end_idx - 1), column_names_to_ignore], 
            time_frame = dataset.loc[start:(end_idx - 1), recording_idx_name],
            activities = dataset.loc[start:(end_idx - 1), label_column_name]
            subject = dataset.iloc[start_idx]['subject']
        ))
        start_idx = end_idx + 1
    # TODO: make this work
    return recordings

