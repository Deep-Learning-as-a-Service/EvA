import os
import pandas as pd
from utils.Recording import Recording

def filter_activities_negative(activities_to_remove: list):
    return filter_activities_custom(lambda activities: ~activities.isin(activities_to_remove))


def filter_activities_custom(
        filter_fn,
        new_idx,
):
    """
    Removes all activities where filter_fn is false
    """

    def fn(recordings: list[Recording]):
        for recording in recordings:
            recording.activities.reset_index(drop=True, inplace=True)
            recording.sensor_frame.reset_index(drop=True, inplace=True)
            recording.time_frame.reset_index(drop=True, inplace=True)

            recording.activities = new_idx(
                recording.activities[filter_fn(recording.activities)])
            recording.sensor_frame = recording.sensor_frame.loc[recording.activities.index]
            recording.time_frame = recording.time_frame.loc[recording.activities.index]

        return recordings

    return fn

def filter_activities(activities_to_keep: list):
    return filter_activities_custom(
        lambda activities: activities.isin(activities_to_keep),
        new_idx=lambda activities: activities)

def load_lab_dataset(path: str, activityLabelToIndexMap: dict, features = None, limit: int = None) -> "list[Recording]":
    """
    Load the recordings from a folder containing csv files.
    """
    recordings = []

    recording_files = os.listdir(path)
    recording_files = list(
        filter(lambda file: file.endswith('.csv'), recording_files))

    if limit is not None:
        recording_files = recording_files[:limit]

    recording_files = sorted(
        recording_files, key=lambda file: int(file.split('_')[0]))

    for (index, file) in enumerate(recording_files):
        print(f'Loading recording {file}, {index+1} / {len(recording_files)}')

        recording_dataframe = pd.read_csv(os.path.join(path, file))
        time_frame = recording_dataframe.loc[:, 'SampleTimeFine']
        # .map(lambda label: activityLabelToIndexMap[label])
        activities = recording_dataframe.loc[:, 'activity']
        sensor_frame = recording_dataframe.loc[:,
                                               recording_dataframe.columns.difference(['SampleTimeFine', 'activity'])]
        if features is not None:
            sensor_frame = sensor_frame[features]

        subject = file.split('_')[1]

        recordings.append(Recording(sensor_frame, time_frame,
                          activities, subject, index))

    print(f'Loaded {len(recordings)} recordings from {path}')
    recordings = filter_activities(
        list(activityLabelToIndexMap.keys()))(recordings)

    for rec in recordings:
        rec.activities = rec.activities.map(
            lambda label: activityLabelToIndexMap[label])
    return recordings