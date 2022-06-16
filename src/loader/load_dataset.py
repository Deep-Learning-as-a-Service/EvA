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
    print(f"Will read dataset from {path}")
    dataset = pd.read_csv(path, header=0, engine='python', verbose=True)

    assert len(dataset.columns) == len(set(dataset.columns)), "Duplicate column names"
    feature_column_names = list(set(list(dataset.columns)) - set([label_column_name, recording_idx_name]) - set(column_names_to_ignore))

    recording_idxs = np.array(dataset.loc[:, recording_idx_name])
    recording_change_idxs = np.where(recording_idxs[:-1] != recording_idxs[1:])[0] + 1
    recording_end_idx = np.append(recording_change_idxs, len(dataset) + 1) # end of last recording

    print(f"convert to Recording objects...")
    start_idx = 0
    recordings = []
    for i, end_idx in enumerate(recording_end_idx):
        recordings.append(Recording(
            sensor_frame = dataset.loc[start_idx:(end_idx - 1), feature_column_names], 
            time_frame = dataset.loc[start_idx:(end_idx - 1), recording_idx_name],
            activities = dataset.loc[start_idx:(end_idx - 1), label_column_name],
            subject = int(dataset.iloc[start_idx]['SUBJECT_IDX']),
            identifier = str(i)
        ))
        start_idx = end_idx + 1
    # TODO: make this work
    return recordings

