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
from sklearn.model_selection import KFold, StratifiedKFold
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
import tensorflow as tf
from utils.window_test_percentages import window_test_percentages


def get_data(load_recordings, shuffle_seed, num_folds):

    # Funcs --------------------------------------------------------------------------------------------------------------

    window_size = settings.data_dimension_dict["window_size"]
    n_classes = settings.data_dimension_dict["n_classes"]
        
    preprocess = lambda recordings: Preprocessor().jens_preprocess(recordings)
    windowize = lambda recordings: Windowizer(window_size=window_size).jens_windowize(recordings)
    convert = lambda windows: Converter(n_classes=n_classes).sonar_convert(windows)
    flatten = lambda tuple_list: [item for sublist in tuple_list for item in sublist]

    recordings = load_recordings()

    # Dirty!!!!! pls refactor we set the global vars for the Normalisation
    print("Calculating mean and variance of whole dataset, and store it (dirty) in the global settings. This can take a while...")
    startTime = datetime.now()
    sensor_frames = tf.constant(np.concatenate(
        [recording.sensor_frame.to_numpy() for recording in recordings], axis=0))
    layer = tf.keras.layers.Normalization(axis=-1)
    layer.adapt(sensor_frames)

    settings.input_distribution_variance = layer.variance
    settings.input_distribution_mean = layer.mean

    endTime = datetime.now()
    print("Time spent for finding mean and variance: ", str(endTime-startTime))


    random.seed(shuffle_seed)
    random.shuffle(recordings)

    # Preprocessing
    recordings = preprocess(recordings)

    # Validation Splits
    k = num_folds
    k_fold = StratifiedKFold(n_splits=k, random_state=None) #Stratified
    recordings_train = np.array(recordings)
    recordings_validation_splits = [(recordings_train[train_idx], recordings_train[val_idx]) for train_idx, val_idx in k_fold.split([recording.sensor_frame for recording in recordings_train], [recording.subject for recording in recordings_train])]
    # Output: [(recordings_train_01, recordings_test_01), (recordings_train_02, recordings_test_02), ...]
    windows_validation_splits = list(map(lambda validation_split: map(windowize, validation_split), recordings_validation_splits))
    # Output: [(windows_train_01, windows_test_01), (windows_train_02, windows_test_02), ...]

    # for i, windows_train_test in enumerate(windows_validation_splits):
    #     windows_train, windows_test = windows_train_test
    #     window_test_percentages(windows_train, windows_test, f"Split {i+1}")

    X_y_validation_splits = list(map(lambda validation_split: tuple(flatten(map(convert, validation_split))), windows_validation_splits))
    # Output: [(X_train_01, y_train_01, X_val_01, y_val_01), (X_train_02, y_train_02, X_val_02, y_val_02), ...]



    return X_y_validation_splits