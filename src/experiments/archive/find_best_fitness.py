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
from utils.Tester import Tester
import utils.settings as settings

from utils.folder_operations import new_saved_experiment_folder
from evaluation.conf_matrix import create_conf_matrix
from evaluation.text_metrics import create_text_metrics
from evaluation.metrics import accuracy, f1_score_

from model_representation.ParametrizedLayer.PDenseLayer import PDenseLayer
from model_representation.ParametrizedLayer.PConv1DLayer import PConv1DLayer
from model_representation.ParametrizedLayer.PConv2DLayer import PConv2DLayer
from optimizer.SeqEvo.EvoTechniqueConfig import DefaultEvoTechniqueConfig
from optimizer.SeqEvo.Selector import Selector
from datetime import datetime
from model_representation.ParametrizedLayer.ParametrizedLayer import ParametrizedLayer
from model_representation.ParametrizedLayer.PLstmLayer import PLstmLayer
from utils.logger import logger
from optimizer.SeqEvo.InitialModelLayer import InitialModelLayer
from loader.get_opportunity_data import get_opportunity_data
from evaluation.Fitness import Fitness
import utils.config as config

testing = False


# Config --------------------------------------------------------------------------

experiment_configs = [ 
                (2, 2, "-1001731938222", "logs/2x2_fold_acc.txt", "logs/2x2_fold_acc_tester.txt", "small_split_kfold_max_val_iter_acc"),
                (3, 3, "-1001555874641", "logs/3x3_fold_acc.txt", "logs/3x3_fold_acc_tester.txt", "small_split_kfold_max_val_iter_acc"),
                (3, 2, "-1001790367792", "logs/3x2_fold_acc.txt", "logs/3x2_fold_acc_tester.txt", "small_split_kfold_max_val_iter_f1"),
                (4, 4, "-1001675856254", "logs/4x4_fold_acc.txt", "logs/4x4_fold_f1_tester.txt", "small_split_kfold_max_val_iter_f1")
            ]
for num_folds, validation_iterations, telegram_chat_id, log_path, tester_path, fitness_func_str in experiment_configs:
    
    # Experiment Name ---------------------------------------------------------------
    experiment_name = f"{num_folds}x{validation_iterations}_fold" 
    currentDT = datetime.now()
    currentDT_str = currentDT.strftime("%y-%m-%d_%H-%M-%S_%f")
    experiment_name = experiment_name + "-" + currentDT_str


    window_size = 30*3
    n_features = 51
    n_classes = 6
    evo_generations = 50
    config.telegram_chat_id = telegram_chat_id

    
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
    tester = Tester(tester_path, X_train, y_train, X_test, y_test)

    # Optimization -------------------------------------------------------------------------------------------------------

    # Create Folder, save model export and evaluations there
    experiment_folder_path = new_saved_experiment_folder(experiment_name) # create folder to store results
    seqevo_history = SeqEvoHistory(
        path_to_file=os.path.join(experiment_folder_path, 'seqevo_history.csv')
    )

    # Config
    parent_selector = Selector.select_from_fitness_probability
    technique_config = DefaultEvoTechniqueConfig()
    fitness_obj = Fitness(X_train, y_train, X_test, y_test, X_y_validation_splits, validation_iterations)
    fitness_func = getattr(fitness_obj, fitness_func_str)
    fitness = fitness_func if not testing else lambda model_genome, log_func: random.random()

    logger_ = lambda *args, **kwargs: logger(*args, path=log_path, **kwargs)
    
    # NAS - Neural Architecture Search
    model_genome = SeqEvo(
        n_generations = evo_generations, 
        pop_size = 10,
        fitness_func = fitness,
        n_parents = 4,
        technique_config = technique_config,
        parent_selector=parent_selector,
        log_func=logger_,
        seqevo_history=seqevo_history,
        initial_models = [],
        tester=tester
    ).run()

raise Exception("**** EVO DONE EXCEPTION *****") # TODO bigger smaller split, evaluate the optimization