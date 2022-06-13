"""
keras exp
"""


import os
import random
import numpy as np
import pandas as pd
from datetime import datetime
from loader.load_dataset import load_dataset
from loader.Preprocessor import Preprocessor
import utils.settings as settings
from utils.array_operations import split_list_by_percentage

from utils.folder_operations import new_saved_experiment_folder
from evaluation.conf_matrix import create_conf_matrix
from evaluation.text_metrics import create_text_metrics
from evaluation.metrics import accuracy, f1_score
from utils.Windowizer import Windowizer
from sklearn.model_selection import KFold
from utils.Converter import Converter

from models.JensModel import JensModel
from models.MultilaneConv import MultilaneConv
from models.BestPerformerConv import BestPerformerConv
from models.OldLSTM import OldLSTM
from models.SenselessDeepConvLSTM import SenselessDeepConvLSTM
from models.LeanderDeepConvLSTM import LeanderDeepConvLSTM
from model_representation.ParametrizedLayer.PConv1DLayer import PConv1DLayer
from model_representation.ParametrizedLayer.PConv2DLayer import PConv2DLayer
from optimizer.SeqEvo.EvoTechniqueConfig import DefaultEvoTechniqueConfig
from optimizer.SeqEvo.Selector import Selector
from optimizer.SeqEvo.Crosser import Crosser
from datetime import datetime
from model_representation.ParametrizedLayer.ParametrizedLayer import ParametrizedLayer
from model_representation.ParametrizedLayer.PLstmLayer import PLstmLayer
from model_representation.ParametrizedLayer.PDenseLayer import PDenseLayer




layer_pool: 'list[ParametrizedLayer]' = [PConv2DLayer, PDenseLayer, PLstmLayer] #PConv1DLayer
settings.init(_layer_pool=layer_pool)

experiment_name = "leander_deep_conv_comparison"

currentDT = datetime.now()
currentDT_str = currentDT.strftime("%y-%m-%d_%H-%M-%S_%f")
experiment_name = experiment_name + "-" + currentDT_str

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


# Config --------------------------------------------------------------------------------------------------------------
window_size = 30*3
n_classes = 6

load_recordings = lambda: load_dataset(os.path.join(settings.opportunity_dataset_csv_path, 'data.csv'), 
    label_column_name='ACTIVITY_IDX', 
    recording_idx_name='RECORDING_IDX', 
    column_names_to_ignore=['SUBJECT_IDX', 'MILLISECONDS']
)

preprocess = lambda recordings: Preprocessor().jens_preprocess_with_normalize(recordings)
windowize = lambda recordings: Windowizer(window_size=window_size).jens_windowize(recordings)
convert = lambda windows: Converter(n_classes=n_classes).sonar_convert(windows)
flatten = lambda tuple_list: [item for sublist in tuple_list for item in sublist]
test_train_split = lambda recordings: leave_person_out_split(test_person_idx=2)(recordings)


# Load data
recordings = load_recordings()

random.seed(1678978086101) # 1678978086101 # 277899747
random.shuffle(recordings)

# Preprocessing
recordings = preprocess(recordings)

# Test Train Split
recordings_train, recordings_test = test_train_split(recordings)

# Windowize
windows_train, windows_test = windowize(recordings_train), windowize(recordings_test)

# Convert
X_train, y_train, X_test, y_test = tuple(flatten(map(convert, [windows_train, windows_test])))

# or JensModel
model = LeanderDeepConvLSTM(
    window_size=window_size, 
    n_features=recordings[0].sensor_frame.shape[1], 
    n_outputs=n_classes, 
    n_epochs=5, 
    learning_rate=0.001, 
    batch_size=32, 
    wandb_config={
        'project': 'all_experiments_project',
        'entity': 'valentindoering',
        'name': experiment_name
    }
)
# learning_rate=0.001
# wandb_project=experiment_name
model.fit(X_train, y_train)

y_test_pred = model.predict(X_test)

# Create Folder, save model export and evaluations there
experiment_folder_path = new_saved_experiment_folder(experiment_name) # create folder to store results

# model.export(experiment_folder_path) # opt: export model to folder
create_conf_matrix(experiment_folder_path, y_test_pred, y_test)
create_text_metrics(experiment_folder_path, y_test_pred, y_test, [accuracy]) # TODO: at the moment only with one function working! data gets changed (passed by reference) - refactor metric functions
