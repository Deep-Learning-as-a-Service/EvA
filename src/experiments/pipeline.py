"""
Our first version of our automated pipeline
"""


import os
import random
from loader.load_dataset import load_dataset
from loader.Preprocessor import Preprocessor
import utils.settings as settings
from utils.array_operations import split_list_by_percentage
from models.JensModel import JensModel
from utils.folder_operations import new_saved_experiment_folder
from evaluation.conf_matrix import create_conf_matrix
from evaluation.text_metrics import create_text_metrics
from evaluation.metrics import accuracy, f1_score


settings.init()
# label (activity idx),context 2 (subject idx),context (recording idx)
# Load data
recordings = load_dataset(os.path.join(settings.opportunity_dataset_csv_path, 'data.csv'), 
    label_column_name='label (activity idx)', 
    recording_idx_name='context (recording idx)', 
    column_names_to_ignore=['context 2 (subject idx)', 'milliseconds']
)