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
from utils.logger import logger as log_func
from optimizer.SeqEvo.InitialModelLayer import InitialModelLayer
from evaluation.Fitness import Fitness
from utils.Tester import Tester
from optimizer.HyPaOptuna.HyPaOptuna import HyPaOptuna
from loader.load_dataset import load_dataset
from loader.get_data import get_data

testing = False

# Experiment Name ---------------------------------------------------------------
experiment_name = "activationnormalisation_artemis"
currentDT = datetime.now()
currentDT_str = currentDT.strftime("%y-%m-%d_%H-%M-%S_%f")
experiment_name = experiment_name + "-" + currentDT_str
logger = lambda *args, **kwargs: log_func(*args, path=f"logs/{experiment_name}", **kwargs)


# Config --------------------------------------------------------------------------
window_size = 30*3
n_features = 51
n_classes = 6
num_folds = 5
validation_iterations = 3

layer_pool: 'list[ParametrizedLayer]' = [PConv2DLayer, PDenseLayer, PLstmLayer] #PConv1DLayer
data_dimension_dict = {
    "window_size": window_size,
    "n_features": n_features,
    "n_classes": n_classes
}
settings.init(_layer_pool=layer_pool, _data_dimension_dict=data_dimension_dict)

load_recs = lambda: load_dataset(os.path.join(settings.opportunity_dataset_csv_path, 'data.csv'), 
    label_column_name='ACTIVITY_IDX', 
    recording_idx_name='RECORDING_IDX', 
    column_names_to_ignore=['SUBJECT_IDX', 'MILLISECONDS']
)

X_y_validation_splits = get_data(
    load_recordings=load_recs,
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
fitness = Fitness([], [], [], [], X_y_validation_splits, None).small_split_kfold_acc if not testing else lambda model_genome, log_func: random.random()

# NAS - Neural Architecture Search
n_generations = 150 if not testing else 2
model_genome = SeqEvo(
    n_generations = n_generations, 
    pop_size = 12,
    fitness_func = fitness,
    n_parents = 4,
    technique_config = technique_config,
    parent_selector=parent_selector,
    log_func=logger,
    seqevo_history=seqevo_history,
    initial_models = [],
    tester=None
).run()

model_genome = HyPaOptuna(
    input_model_genome=model_genome,
    n_trials=100,
    fitness_func=fitness,
    log_func=logger
).run()

raise Exception("**** EVO DONE EXCEPTION *****") # TODO bigger smaller split, evaluate the optimization

