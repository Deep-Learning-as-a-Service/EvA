import os
import random
import numpy as np
import pandas as pd
from loader.load_dataset import load_dataset
from loader.Preprocessor import Preprocessor
from optimizer.SeqEvo.SeqEvo import SeqEvo
from optimizer.SeqEvo.SeqEvoHistory import SeqEvoHistory
import utils.settings as settings
from utils.array_operations import split_list_by_percentage
from tensorflow import keras

from utils.folder_operations import new_saved_experiment_folder
from evaluation.conf_matrix import create_conf_matrix
from evaluation.text_metrics import create_text_metrics
from evaluation.metrics import accuracy, f1_score
from utils.Windowizer import Windowizer
from sklearn.model_selection import KFold
from utils.Converter import Converter
from optimizer.NeatNAS.NeatNAS import NeatNAS
from model_representation.ParametrizedLayer.PDenseLayer import PDenseLayer
from model_representation.EvoParam.IntEvoParam import IntEvoParam
from model_representation.ParametrizedLayer.PConv1DLayer import PConv1DLayer
from model_representation.ParametrizedLayer.PConv2DLayer import PConv2DLayer
from optimizer.SeqEvo.EvoTechniqueConfig import DefaultEvoTechniqueConfig
from optimizer.SeqEvo.Selector import Selector
from optimizer.SeqEvo.Crosser import Crosser
from datetime import datetime
from model_representation.ParametrizedLayer.ParametrizedLayer import ParametrizedLayer
from model_representation.ParametrizedLayer.PLstmLayer import PLstmLayer
from utils.progress_bar import print_progress_bar
from utils.logger import logger
from optimizer.SeqEvo.InitialModelLayer import InitialModelLayer

def get_opportunity_data(window_size, n_features, n_classes):

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

    load_recordings = lambda: load_dataset(os.path.join(settings.opportunity_dataset_csv_path, 'data_small.csv'), 
        label_column_name='ACTIVITY_IDX', 
        recording_idx_name='RECORDING_IDX', 
        column_names_to_ignore=['SUBJECT_IDX', 'MILLISECONDS']
    )


    preprocess = lambda recordings: Preprocessor().jens_preprocess(recordings)
    windowize = lambda recordings: Windowizer(window_size=window_size).jens_windowize(recordings)
    convert = lambda windows: Converter(n_classes=n_classes).sonar_convert(windows)
    flatten = lambda tuple_list: [item for sublist in tuple_list for item in sublist]
    test_train_split = lambda recordings: leave_recording_out_split(test_percentage=0.25)(recordings)

    recordings = load_recordings()

    random.seed(1678978086101)
    random.shuffle(recordings)

    # Preprocessing
    recordings = preprocess(recordings)

    # Test Train Splits ----------------------------------------------------------------------------------------------------
    recordings_train, recordings_test = test_train_split(recordings)

    # Validation Splits
    k = 2
    k_fold = KFold(n_splits=k, random_state=None)
    recordings_train = np.array(recordings_train)
    recordings_validation_splits = [(recordings_train[train_idx], recordings_train[val_idx]) for train_idx, val_idx in k_fold.split(recordings_train)]
    # Output: [(recordings_train_01, recordings_test_01), (recordings_train_02, recordings_test_02), ...]
    windows_validation_splits = list(map(lambda validation_split: map(windowize, validation_split), recordings_validation_splits))
    # Output: [(windows_train_01, windows_test_01), (windows_train_02, windows_test_02), ...]
    X_y_validation_splits = list(map(lambda validation_split: tuple(flatten(map(convert, validation_split))), windows_validation_splits))
    # Output: [(X_train_01, y_train_01, X_val_01, y_val_01), (X_train_02, y_train_02, X_val_02, y_val_02), ...]


    # Windowize, Convert --------------------------------------------------------------------------------------------------
    windows_train, windows_test = windowize(recordings_train), windowize(recordings_test)
    X_train, y_train, X_test, y_test = tuple(flatten(map(convert, [windows_train, windows_test])))

    return X_train, y_train, X_test, y_test, X_y_validation_splits



