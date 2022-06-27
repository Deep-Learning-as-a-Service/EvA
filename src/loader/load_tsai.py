
import pandas as pd
from utils.Recording import Recording
from tsai.all import get_classification_data
import random
import numpy as np
from sklearn.model_selection import KFold
from loader.Preprocessor import Preprocessor
from utils.Converter import Converter
from utils.Window import Window
from utils.Recording import Recording
from utils.array_operations import split_list_by_percentage
import utils.settings as settings


def load_tsai(
    shuffle_seed:int,
    num_folds:int):
    
    leave_recording_out_split = lambda test_percentage: lambda recordings: split_list_by_percentage(list_to_split=recordings, percentage_to_split=test_percentage)

    # Funcs --------------------------------------------------------------------------------------------------------------

    def recording_to_window(recording: Recording):
        return Window(
            sensor_array=recording.sensor_frame.to_numpy(),
            activity = int(recording.activities[0]),
            subject = -1
        )

    n_classes = settings.data_dimension_dict["n_classes"]
        
    preprocess = lambda recordings: Preprocessor().jens_preprocess_with_normalize(recordings)
    convert = lambda windows: Converter(n_classes=n_classes).sonar_convert(windows)
    flatten = lambda tuple_list: [item for sublist in tuple_list for item in sublist]
    test_train_split = lambda recordings: leave_recording_out_split(test_percentage=0.3)(recordings)


    X,y,splits = get_classification_data('LSST', split_data=False)
    X = np.swapaxes(X, 1, 2)

    recordings = []
    for idx, x_sample in enumerate(X):
        sensor_f = pd.DataFrame(data= x_sample)
        recordings.append(
            Recording(
                sensor_frame=sensor_f,
                time_frame=None,
                activities=pd.Series(np.repeat(settings.tsai_initial_num_to_activity_idx[int(y[idx])], len(sensor_f))),
                subject=None,
                identifier=idx
            )
        )

    random.seed(shuffle_seed)
    random.shuffle(recordings)
    recordings = preprocess(recordings)

    recordings_train, recordings_test = test_train_split(recordings)

    k = num_folds
    k_fold = KFold(n_splits=k, random_state=None)
    recordings_train = np.array(recordings_train)
    recordings_validation_splits = [(recordings_train[train_idx], recordings_train[val_idx]) for train_idx, val_idx in k_fold.split([recording.sensor_frame for recording in recordings_train])]
    windows_validation_splits = []
    for idx in range(len(recordings_validation_splits)):
        windows_validation_splits.append((list(map(recording_to_window, recordings_validation_splits[idx][0])), list(map(recording_to_window, recordings_validation_splits[idx][1]))))
    X_y_validation_splits = list(map(lambda validation_split: tuple(flatten(map(convert, validation_split))), windows_validation_splits))
    windows_train, windows_test = list(map(recording_to_window, recordings_train)), list(map(recording_to_window, recordings_test))
    X_train, y_train, X_test, y_test = tuple(flatten(map(convert, [windows_train, windows_test])))

    return X_train, y_train, X_test, y_test, X_y_validation_splits
