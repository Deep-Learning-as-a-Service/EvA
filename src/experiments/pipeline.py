"""
Our first version of our automated pipeline
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
from utils.folder_operations import new_saved_experiment_folder
from evaluation.conf_matrix import create_conf_matrix
from evaluation.text_metrics import create_text_metrics
from evaluation.metrics import accuracy, f1_score
from utils.Windowizer import Windowizer
from sklearn.model_selection import KFold
from utils.Converter import Converter


settings.init()

# Load data
recordings = load_dataset(os.path.join(settings.opportunity_dataset_csv_path, 'data.csv'), 
    label_column_name='ACTIVITY_IDX', 
    recording_idx_name='RECORDING_IDX', 
    column_names_to_ignore=['SUBJECT_IDX', 'MILLISECONDS']
)

random.seed(1678978086101)
random.shuffle(recordings)

# Preprocessing
recordings = Preprocessor().jens_preprocess(recordings)

# Test Train Split
test_percentage = 0.3
recordings_train, recordings_test = split_list_by_percentage(recordings, test_percentage)

# Functions
windowize = Windowizer(window_size=25).jens_windowize
convert = Converter(n_classes=6).jens_convert
flatten = lambda tuple_list: [item for sublist in tuple_list for item in sublist]

# Validation Splits
k = 4
k_fold = KFold(n_splits=k, random_state=None)
recordings_train = np.array(recordings_train)
recordings_validation_splits = [(recordings_train[train_idx], recordings_train[val_idx]) for train_idx, val_idx in k_fold.split(recordings_train)]
# Output: [(recordings_train_01, recordings_test_01), (recordings_train_02, recordings_test_02), ...]
windows_validation_splits = list(map(lambda validation_split: map(windowize, validation_split), recordings_validation_splits))
# Output: [(windows_train_01, windows_test_01), (windows_train_02, windows_test_02), ...]
X_y_validation_splits = list(map(lambda validation_split: tuple(flatten(map(convert, validation_split))), windows_validation_splits))
# Output: [(X_train_01, y_train_01, X_val_01, y_val_01), (X_train_02, y_train_02, X_val_02, y_val_02), ...]

windows_train, windows_test = windowize(recordings_train), windowize(recordings_test)

# Convert
X_train, y_train, X_test, y_test = tuple(flatten(map(convert, [windows_train, windows_test])))


def fitness(model_genome) -> float:
    model = ModelBuilder(model_genome)

    # Traininsparams
    batch_size = model_genome.batch_size
    learning_rate = model_genome.learning_rate
    n_epochs = model_genome.n_epochs

    accuracies = []
    for idx, (X_train, y_train, X_val, y_val) in enumerate(X_y_validation_splits):
        print(f"Doing fold {idx}/{k-1} ... ============================================================")
        model.fit(X_train, y_train, batch_size, learning_rate, n_epochs)
        y_val_pred = model.predict(X_val)
        acccuracies.append(accuracy(y_val, y_val_pred))
    return np.mean(accuracies)

# NAS - Neural Architecture Search
model_genome = NeatNAS(n_generation = 1, population_size = 10).run(fitness)

# Find Architecture Params
dna = DNA(params_to_optimmize)
model_genome = ArchitectureParamOptimizer(n_evaluations = 100, dna = dna).run(model_genome, fitness)

# Find Training Params
model_genome = Training_Params(model_genome, fitness)

# Test, Evaluate
model = ModelBuilder(model_genome)
y_test_pred = model.predict(X_test)

# Create Folder, save model export and evaluations there
experiment_folder_path = new_saved_experiment_folder('opportunity_jens_cnn') # create folder to store results

# model.export(experiment_folder_path) # opt: export model to folder
create_conf_matrix(experiment_folder_path, y_test_pred, y_test)
create_text_metrics(experiment_folder_path, y_test_pred, y_test, [accuracy]) # TODO: at the moment only with one function working! data gets changed (passed by reference) - refactor metric functions
