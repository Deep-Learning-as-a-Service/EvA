"""
Our first version of our automated pipeline that is sequential only and uses our own evolutionary algorithm
"""


import os
import random
import numpy as np
import pandas as pd
from loader.load_dataset import load_dataset
from loader.Preprocessor import Preprocessor
from optimizer.SeqEvo.SeqEvo import SeqEvo
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
from optimizer.SeqEvo.Selector import Selector
from optimizer.SeqEvo.Crosser import Crosser
from datetime import datetime
from model_representation.ParametrizedLayer.ParametrizedLayer import ParametrizedLayer
from model_representation.ParametrizedLayer.PLstmLayer import PLstmLayer
from utils.progress_bar import print_progress_bar

# Experiment Name ---------------------------------------------------------------
experiment_name = "best-performer-conv-finally"
currentDT = datetime.now()
currentDT_str = currentDT.strftime("%y-%m-%d_%H-%M-%S_%f")
experiment_name = experiment_name + "-" + currentDT_str

# Config --------------------------------------------------------------------------
window_size = 30*3
n_features = 51
n_classes = 6

layer_pool: 'list[ParametrizedLayer]' = [PConv2DLayer, PDenseLayer, PLstmLayer] #PConv1DLayer
settings.init(_layer_pool=layer_pool)


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

load_recordings = lambda: load_dataset(os.path.join(settings.opportunity_dataset_csv_path, 'data.csv'), 
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
test_percentage = 0.3
recordings_train, recordings_test = test_train_split(recordings)

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


# Windowize, Convert --------------------------------------------------------------------------------------------------
windows_train, windows_test = windowize(recordings_train), windowize(recordings_test)
X_train, y_train, X_test, y_test = tuple(flatten(map(convert, [windows_train, windows_test])))


# Fitness Funcs ------------------------------------------------------------------------------------------------------
def fitness_val_split(model_genome, log_func=print) -> float:
    prog_bar = lambda progress: print_progress_bar(progress, total=len(X_y_validation_splits), prefix="k_fold", suffix=f"{progress}/{len(X_y_validation_splits)}", length=30, log_func=log_func, fill=">")
    # Refactoring idea
    # model_genome.fit(X_train, y_train)

    # Traininsparams
    batch_size = model_genome.batch_size
    learning_rate = model_genome.learning_rate
    n_epochs = model_genome.n_epochs

    accuracies = []
    for idx, (X_train, y_train, X_val, y_val) in enumerate(X_y_validation_splits):
        prog_bar(progress=idx)
        model = model_genome.get_model(
            window_size=window_size,
            n_features=n_features,
            n_classes=n_classes
        )
        model.fit(
            X_train, 
            y_train, 
            batch_size=model_genome.batch_size, 
            epochs=model_genome.n_epochs,
            verbose=0
        )
        y_val_pred = model.predict(X_val)
        accuracies.append(accuracy(y_val, y_val_pred))

    prog_bar(progress=len(X_y_validation_splits))

    fitness = np.mean(accuracies)
    return fitness

def fitness_easy(model_genome, log_func=print) -> float:
    model = model_genome.get_model(
        window_size=window_size,
        n_features=n_features,
        n_classes=n_classes
    )
    model.fit(
        X_train, 
        y_train, 
        batch_size=model_genome.batch_size, 
        epochs=model_genome.n_epochs,
        verbose=0
    )

    y_test_pred = model.predict(X_test)
    fitness = accuracy(y_test, y_test_pred)
    return fitness

# Optimization -------------------------------------------------------------------------------------------------------

# Config
parent_selector = Selector.select_from_fitness_probability
crossover_func = Crosser.middlepoint_crossover
generation_distribution = {
    "crossover" : 4,
    "mutate_low" : 4,
    "mutate_mid" : 4,
    "mutate_high" : 4,
    "mutate_all" : 4
}

def log_func(*args, **kwargs):
    message = args[0] if len(args) > 0 else ""
    print(message)
    with open("logs.txt", "a+") as f:
        f.write(message + "\n")

# NAS - Neural Architecture Search
model_genome = SeqEvo(
    n_generations = 500, 
    pop_size = 20,
    fitness_func = fitness_val_split,
    n_parents = 4,
    generation_distribution = generation_distribution,
    parent_selector=parent_selector,
    crossover_func=crossover_func,
    log_func=log_func
).run()


raise Exception("Done") # TODO bigger smaller split, evaluate the optimization

# Find Architecture Params
dna = DNA(params_to_optimmize)
model_genome = ArchitectureParamOptimizer(n_evaluations = 100, dna = dna).run(model_genome, fitness)

# Find Training Params
model_genome = Training_Params(model_genome, fitness)

# Test, Evaluate
model = ModelBuilder(model_genome)
y_test_pred = model.predict(X_test)

# Create Folder, save model export and evaluations there
experiment_folder_path = new_saved_experiment_folder('nas_first_impl') # create folder to store results

# model.export(experiment_folder_path) # opt: export model to folder
create_conf_matrix(experiment_folder_path, y_test_pred, y_test)
create_text_metrics(experiment_folder_path, y_test_pred, y_test, [accuracy]) # TODO: at the moment only with one function working! data gets changed (passed by reference) - refactor metric functions
