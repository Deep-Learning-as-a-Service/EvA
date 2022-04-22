"""
keras exp
"""


import os
import random
import numpy as np
import pandas as pd
from loader.load_dataset import load_dataset
from loader.Preprocessor import Preprocessor
import utils.settings as settings
from utils.array_operations import split_list_by_percentage
from models.JensModel import JensModel
from models.SimpleModel import SimpleModel
from utils.folder_operations import new_saved_experiment_folder
from evaluation.conf_matrix import create_conf_matrix
from evaluation.text_metrics import create_text_metrics
from evaluation.metrics import accuracy, f1_score
from utils.Windowizer import Windowizer
from sklearn.model_selection import KFold
from utils.Converter import Converter


settings.init()

# Configuration
window_size = 25
load_recordings = lambda: load_dataset(os.path.join(settings.opportunity_dataset_csv_path, 'data.csv'), 
    label_column_name='ACTIVITY_IDX', 
    recording_idx_name='RECORDING_IDX', 
    column_names_to_ignore=['SUBJECT_IDX', 'MILLISECONDS']
)
preprocess = lambda recordings: Preprocessor().jens_preprocess_with_normalize(recordings)
windowize = lambda recordings: Windowizer(window_size=window_size).jens_windowize(recordings)
convert = lambda windows: Converter(n_classes=6).jens_convert(windows)
flatten = lambda tuple_list: [item for sublist in tuple_list for item in sublist]

# Load data
recordings = load_recordings()

random.seed(1678978086101)
random.shuffle(recordings)

# Preprocessing
recordings = preprocess(recordings)
# Test Train Split
test_percentage = 0.3
recordings_train, recordings_test = split_list_by_percentage(recordings, test_percentage)

# Windowize
windows_train, windows_test = windowize(recordings_train), windowize(recordings_test)

# Convert
X_train, y_train, X_test, y_test = tuple(flatten(map(convert, [windows_train, windows_test])))

# or JensModel
model = ExtendedModel(window_size=window_size, n_features=recordings[0].sensor_frame.shape[1], n_outputs=6, verbose=1, n_epochs=10)
model.fit(X_train, y_train)

y_test_pred = model.predict(X_test)

# Create Folder, save model export and evaluations there
experiment_folder_path = new_saved_experiment_folder('extended_model') # create folder to store results

# model.export(experiment_folder_path) # opt: export model to folder
create_conf_matrix(experiment_folder_path, y_test_pred, y_test)
create_text_metrics(experiment_folder_path, y_test_pred, y_test, [accuracy]) # TODO: at the moment only with one function working! data gets changed (passed by reference) - refactor metric functions
