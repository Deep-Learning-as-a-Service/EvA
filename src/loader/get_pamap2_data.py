
import os
import random
import numpy as np

from sklearn.model_selection import StratifiedKFold, KFold
from loader.Preprocessor import Preprocessor
from loader.load_pamap2 import load_pamap2_dataset
from utils.Converter import Converter
from utils.Windowizer import Windowizer
from utils.array_operations import split_list_by_percentage
import utils.settings as settings


def get_pamap2_data(        
        shuffle_seed: int,
        num_folds: int):
    # Lib -----------------------------------------------------------
    leave_recording_out_split = lambda test_percentage: lambda recordings: split_list_by_percentage(list_to_split=recordings, percentage_to_split=test_percentage)
    # leave_recording_out_split(test_percentage=0.3)(recordings)
    def leave_person_out_split_idx(recordings, test_person_idx):
        subset_from_condition = lambda condition, recordings: [recording for recording in recordings if condition(recording)] 
        recordings_train = subset_from_condition(lambda recording: recording.subject != test_person_idx, recordings)
        recordings_test = subset_from_condition(lambda recording: recording.subject == test_person_idx, recordings)
        return recordings_train, recordings_test
    leave_person_out_split = lambda test_person_idx: lambda recordings: leave_person_out_split_idx(recordings=recordings, test_person_idx=test_person_idx)
    # leave_person_out_split(test_person_idx=2)(recordings) # 1-4, TODO: could be random


    # Funcs --------------------------------------------------------------------------------------------------------------

    load_recordings = lambda: load_pamap2_dataset()

    window_size = settings.data_dimension_dict["window_size"]
    n_classes = settings.data_dimension_dict["n_classes"]
        
    preprocess = lambda recordings: Preprocessor().jens_preprocess_with_normalize(recordings)
    windowize = lambda recordings: Windowizer(window_size=window_size).jens_windowize(recordings)
    convert = lambda windows: Converter(n_classes=n_classes).sonar_convert(windows)
    flatten = lambda tuple_list: [item for sublist in tuple_list for item in sublist]
    test_train_split = lambda recordings: leave_recording_out_split(test_percentage=0.3)(recordings)
    # test_train_split = lambda recordings: leave_person_out_split(test_person_idx=2)(recordings)

    recordings = load_recordings()

    random.seed(shuffle_seed)
    random.shuffle(recordings)

    # Preprocessing
    recordings = preprocess(recordings)

    # Test Train Splits ----------------------------------------------------------------------------------------------------
    recordings_train, recordings_test = test_train_split(recordings)

    # Validation Splits
    k = num_folds
    k_fold = KFold(n_splits=k, random_state=None)
    recordings_train = np.array(recordings_train)
    recordings_validation_splits = [(recordings_train[train_idx], recordings_train[val_idx]) for train_idx, val_idx in k_fold.split([recording.sensor_frame for recording in recordings_train])]
    # Output: [(recordings_train_01, recordings_test_01), (recordings_train_02, recordings_test_02), ...]
    windows_validation_splits = list(map(lambda validation_split: map(windowize, validation_split), recordings_validation_splits))
    # Output: [(windows_train_01, windows_test_01), (windows_train_02, windows_test_02), ...]

    # for i, windows_train_test in enumerate(windows_validation_splits):
    #     windows_train, windows_test = windows_train_test
    #     window_test_percentages(windows_train, windows_test, f"Split {i+1}")

    X_y_validation_splits = list(map(lambda validation_split: tuple(flatten(map(convert, validation_split))), windows_validation_splits))
    # Output: [(X_train_01, y_train_01, X_val_01, y_val_01), (X_train_02, y_train_02, X_val_02, y_val_02), ...]

    


    # Windowize, Convert --------------------------------------------------------------------------------------------------
    windows_train, windows_test = windowize(recordings_train), windowize(recordings_test)
    # window_test_percentages(windows_train, windows_test, "Big split")
    X_train, y_train, X_test, y_test = tuple(flatten(map(convert, [windows_train, windows_test])))

    return X_train, y_train, X_test, y_test, X_y_validation_splits
