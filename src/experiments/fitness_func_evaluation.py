"""
func evaluation
- a small split fitness func is good, if its result is close to the big split fitness
    - k_fold with carefull split
    - less epochs
    - take best, take worst with faster learning rate
- execute for differnt numbers of folds
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
from loader.get_opportunity_data import get_opportunity_data
from evaluation.Fitness import Fitness

testing = True
test_fitness_big_split = lambda model_genome, logger: 0.6
test_fitness_small_split = lambda model_genome, logger: 0.75

# Experiment Name ---------------------------------------------------------------
experiment_name = "fitness_func_evaluation"
currentDT = datetime.now()
currentDT_str = currentDT.strftime("%y-%m-%d_%H-%M-%S_%f")
experiment_name = experiment_name + "-" + currentDT_str

# Config --------------------------------------------------------------------------
window_size = 30*3
n_features = 51
n_classes = 6
num_folds = 2

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

fi = Fitness(X_train, y_train, X_test, y_test, X_y_validation_splits, verbose=True)
fi.log_split_insights()


logger(f"Starting Experiment fitness_func_evaluation")
# Create Folder, save model export and evaluations there
experiment_folder_path = new_saved_experiment_folder(experiment_name) # create folder to store results
model_genomes_to_check = [
    ('triple_2dconv_lstm_1', InitialModelLayer.triple_2dconv_lstm_1()),
    ('triple_2dconv_lstm_2', InitialModelLayer.triple_2dconv_lstm_2()),
    ('double_2dconv_dense_1', InitialModelLayer.double_2dconv_dense_1()), 
    ('double_2dconv_dense_2', InitialModelLayer.double_2dconv_dense_2()), 
    ('leander_deep_conv_1', InitialModelLayer.leander_deep_conv_1()), 
    ('leander_deep_conv_2', InitialModelLayer.leander_deep_conv_2()), 
    ('jens_1', InitialModelLayer.jens_1()), 
    ('conv_lstm_1', InitialModelLayer.conv_lstm_1())
]

# Printing
stringify_list = lambda l: list(map(str, l))
nl = '\n'

logger(f"Models for comparison {nl}" + nl.join(name for name, _ in model_genomes_to_check))

# Big Split -------------------------------------------------------------------------------------------------------
logger(f"{nl}Starting big_split evaluation ...{nl}")

big_split_evaluation = []
# [('triple_2dconv_lstm_1', InitialModelLayer.triple_2dconv_lstm_1(), [acc_01, acc_02], mean_acc), ...]
big_split_fitness_func = fi.big_split_acc if not testing else test_fitness_big_split
idx = 0
for name, model_genome in model_genomes_to_check:
    model_log = f"Model {idx}/{len(big_split_evaluation)}:'{name}' big_split_fitness"
    logger(f"{model_log} in progress ...")
    fitnesses = [big_split_fitness_func(model_genome, logger) for _ in range(2)]
    mean = np.mean(fitnesses)
    big_split_evaluation.append((name, model_genome, fitnesses, mean))
    logger(f"{model_log} done -> fitnesses: [{' '.join(stringify_list(fitnesses))}], mean: {mean}{nl}")
    idx += 1

# mean sort
big_split_evaluation = sorted(big_split_evaluation, key=lambda big_split_evaluation : big_split_evaluation[-1])

logger(f"Big split fitness done!{nl}")
get_ranking_str = lambda name, fitnesses, mean: f"{name} fitnesses [{' '.join(stringify_list(fitnesses))}] mean: {mean}"
logger(f"Ranking:{nl}")
logger(nl.join([get_ranking_str(name, fitnesses, mean) for name, _, fitnesses, mean in big_split_evaluation]))

# Small Split Fitness Funcs --------------------------------------------------------
logger(f"{nl}Starting small_split evaluation{nl}")

small_split_fitness_funcs = [
    ("small_split_kfold_acc", fi.small_split_kfold_acc),
    ("small_split_kfold_f1", fi.small_split_kfold_acc)
] if not testing else [("small_split_test_func", test_fitness_small_split)]
funcs_to_evaluate_str = " ".join([name for name, _ in small_split_fitness_funcs])
logger(f"{nl}fitness funcs to evaluate [{funcs_to_evaluate_str}]{nl}")

for name, func in small_split_fitness_funcs:
    func_evaluation = []
    # [('triple_2dconv_lstm_1', big_split_fitnesses, big_split_mean_acc, small_split_fitnesses, small_split_mean_acc), ...]
    fitness_func_str = f"fitness func {name}"
    logger(f"{nl}{fitness_func_str} start evaluating ============================================================================{nl}")

    for model_genome_name, model_genome, fitnesses, mean in big_split_evaluation:
        logger(f"{fitness_func_str} on model {model_genome_name} ...")
        func_fitnesses = [func(model_genome, logger) for _ in range(2)]
        func_mean = np.mean(func_fitnesses)
        logger(f"{fitness_func_str} on model {model_genome_name} evaluating done | fitnesses [{' '.join(stringify_list(func_fitnesses))}]{nl}")
        func_evaluation.append((model_genome_name, fitnesses, mean, func_fitnesses, func_mean))
    
    logger(f"{fitness_func_str} FINISHED evaluating!{nl}")

    # Ranking
    logger(f"Big Split Ranking small split fitnesses added{nl}")
    get_ranking_str = lambda model_name, b_s_fitnesses, b_s_mean, s_s_fitnesses, s_s_mean: f"*** {model_name} b_s_fitnesses [{' '.join(stringify_list(b_s_fitnesses))}] mean: {b_s_mean} | s_s_fitnesses [{' '.join(stringify_list(s_s_fitnesses))}] | s_s_mean {s_s_mean}"
    logger('\n'.join([get_ranking_str(model_name, b_s_fitnesses, b_s_mean, s_s_fitnesses, s_s_mean) for model_name, b_s_fitnesses, b_s_mean, s_s_fitnesses, s_s_mean in func_evaluation]))

    # Differences
    mean_differences = [(model_name, round(b_s_mean - s_s_mean, 4)) for model_name, _, b_s_mean, _, s_s_mean in func_evaluation]
    logger(f"{nl}Differences to mean [{nl}{nl.join([f'{model_name} diff {difference}' for model_name, difference in mean_differences])}{nl}]{nl}")
    logger(f"Keep in mind: there can be a differences, thats not the problem, it needs to be constant! (represenative) should not randomly prefer wrong architectures!")

    abs_mean_difference = np.mean(list(map(lambda x: abs(x[1]), mean_differences)))
    logger(f"{nl}Abs Mean difference {abs_mean_difference} (not representative)")


    
    
