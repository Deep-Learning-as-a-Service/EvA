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
from optimizer.SeqEvo.SeqEvoHistory import SeqEvoHistory
import utils.settings as settings
from utils.array_operations import split_list_by_percentage
from tensorflow import keras

from utils.folder_operations import new_saved_experiment_folder
from evaluation.conf_matrix import create_conf_matrix
from evaluation.text_metrics import create_text_metrics
from evaluation.metrics import accuracy, f1_score_
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
from loader.get_opportunity_data import get_opportunity_data
from evaluation.Fitness import Fitness
from utils.window_test_percentages import window_test_percentages

# Experiment Name ---------------------------------------------------------------
experiment_name = "four_fold_f1" # CHAT_ID: -1001790367792
currentDT = datetime.now()
currentDT_str = currentDT.strftime("%y-%m-%d_%H-%M-%S_%f")
experiment_name = experiment_name + "-" + currentDT_str

# Config --------------------------------------------------------------------------
window_size = 30*3
n_features = 51
n_classes = 6
num_folds = 4
validation_iterations = 4

layer_pool: 'list[ParametrizedLayer]' = [PConv2DLayer, PDenseLayer, PLstmLayer] #PConv1DLayer
data_dimension_dict = {
    "window_size": window_size,
    "n_features": n_features,
    "n_classes": n_classes
}
settings.init(_layer_pool=layer_pool, _data_dimension_dict=data_dimension_dict)

X_train, y_train, X_test, y_test, X_y_validation_splits = get_opportunity_data(
    shuffle_seed=1678978086101,
    num_folds=num_folds
)
# Optimization -------------------------------------------------------------------------------------------------------

# Create Folder, save model export and evaluations there
experiment_folder_path = new_saved_experiment_folder(experiment_name) # create folder to store results
seqevo_history = SeqEvoHistory(
    path_to_file=os.path.join(experiment_folder_path, 'seqevo_history.csv')
)

# Config
parent_selector = Selector.select_from_fitness_probability
technique_config = DefaultEvoTechniqueConfig()
fitness = Fitness(X_train, y_train, X_test, y_test, X_y_validation_splits, validation_iterations).kfold_without_test_set_f1
# lambda model_genome, log_func: Fitness(X_train, y_train, X_test, y_test, X_y_validation_splits).normal_with_test_set(model_genome, log_func) # kfold_without_test_set

# NAS - Neural Architecture Search
model_genome = SeqEvo(
    n_generations = 300, 
    pop_size = 10,
    fitness_func = fitness,
    n_parents = 4,
    technique_config = technique_config,
    parent_selector=parent_selector,
    log_func=logger,
    seqevo_history=seqevo_history,
    initial_models = InitialModelLayer.get_all_models()
).run()

# Test, Evaluate
model = model_genome.get_model()

model.fit(
    X_train, 
    y_train, 
    batch_size=model_genome.batch_size, 
    epochs=model_genome.n_epochs,
    verbose=0
)
y_test_pred = model.predict(X_test)

# model.export(experiment_folder_path) # opt: export model to folder
create_conf_matrix(experiment_folder_path, y_test_pred, y_test)
create_text_metrics(experiment_folder_path, y_test_pred, y_test, [accuracy]) # TODO: at the moment only with one function working! data gets changed (passed by reference) - refactor metric functions

raise Exception("**** EVO DONE EXCEPTION *****") # TODO bigger smaller split, evaluate the optimization