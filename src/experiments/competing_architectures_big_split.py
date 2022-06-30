"""
State-of-the-art, Standard models, what is currently possible on the dataset
"""


import os
import random
import numpy as np
import pandas as pd
from datetime import datetime
from loader.load_dataset import load_dataset
from loader.get_opportunity_data import get_opportunity_data_big_split
from loader.Preprocessor import Preprocessor
import utils.settings as settings
from utils.array_operations import split_list_by_percentage


from utils.folder_operations import new_saved_experiment_folder
from evaluation.conf_matrix import create_conf_matrix
from evaluation.text_metrics import create_text_metrics
from utils.Windowizer import Windowizer
from sklearn.model_selection import KFold
from utils.Converter import Converter

from models.JensModel import JensModel
from models.MultilaneConv import MultilaneConv
from models.BestPerformerConv import BestPerformerConv
from models.OldLSTM import OldLSTM
from models.SenselessDeepConvLSTM import SenselessDeepConvLSTM

from model_representation.ParametrizedLayer.PDenseLayer import PDenseLayer
from model_representation.ParametrizedLayer.PConv1DLayer import PConv1DLayer
from model_representation.ParametrizedLayer.PConv2DLayer import PConv2DLayer
from model_representation.ParametrizedLayer.PLstmLayer import PLstmLayer

from models.JensModel import JensModel
from models.MultilaneConv import MultilaneConv
from models.BestPerformerConv import BestPerformerConv
from models.OldLSTM import OldLSTM
from models.MultilaneConvLSTM import MultilaneConvLSTM
from models.KirillAlexDeepConvLSTM import KirillAlexDeepConvLSTM
from models.AlternativeDeepConvLSTM import AlternativeDeepConvLSTM
from models.SenselessDeepConvLSTM import SenselessDeepConvLSTM
from models.LeanderDeepConvLSTM import LeanderDeepConvLSTM
from models.ShallowDeepConvLSTM import ShallowDeepConvLSTM
from sklearn.metrics import f1_score
import random
from utils.seed import set_global_determinism
from optimizer.SeqEvo.InitialModelLayer import InitialModelLayer
from evaluation.Fitness import Fitness
from optimizer.SeqEvo.SeqEvo import SeqEvo
from optimizer.SeqEvo.SeqEvoGenome import SeqEvoGenome
from model_representation.ModelGenome.SeqEvoModelGenome import SeqEvoModelGenome

experiment_name = "competing_architectures_big_split"

currentDT = datetime.now()
currentDT_str = currentDT.strftime("%y-%m-%d_%H-%M-%S_%f")
experiment_name = experiment_name + "-" + currentDT_str

window_size = 30 * 3
n_classes = 6
n_features = 51

layer_pool: 'list[ParametrizedLayer]' = [PConv2DLayer, PDenseLayer, PLstmLayer] #PConv1DLayer
data_dimension_dict = {
    "window_size": window_size,
    "n_features": n_features,
    "n_classes": n_classes
}
settings.init(_layer_pool=layer_pool, _data_dimension_dict=data_dimension_dict)

# Data

X_y_validation_splits = get_opportunity_data_big_split(
    shuffle_seed=1678978086101,
    num_folds=5
)

# models
model_classes = [(JensModel, InitialModelLayer.jens_1())] # (LeanderDeepConvLSTM, InitialModelLayer.leander_deep_conv_1())

for model_class, model_genome in model_classes:
    model_name = model_class.__name__



    # Rainbow Model
    f1_scores = []
    accuracies = []
    for X_train, y_train, X_test, y_test in X_y_validation_splits:
    
        # or JensModel
        model = model_class(
            window_size=window_size, 
            n_features=n_features, 
            n_outputs=n_classes, 
            n_epochs=5, 
            learning_rate=0.001, 
            batch_size=32
        )

        model.fit(X_train, y_train)

        y_test_pred = model.predict(X_test)
        y_test_pred_numbers = np.argmax(y_test_pred, axis=1)
        y_test_numbers = np.argmax(y_test, axis=1)

        # Create Folder, save model export and evaluations there
        # experiment_folder_path = new_saved_experiment_folder(experiment_name) # create folder to store results

        # model.export(experiment_folder_path) # opt: export model to folder
        # create_conf_matrix(experiment_folder_path, y_test_pred, y_test, model_name)
        f1 = f1_score(y_true=y_test_numbers, y_pred=y_test_pred_numbers, average="weighted")
        acc = np.sum(y_test_pred_numbers == y_test_numbers) / len(y_test_numbers)

        print("fold F1:", f1)
        print("fold Acc:", acc)
        f1_scores.append(f1)
        accuracies.append(acc)
    
    print("F1:", np.mean(f1_scores))
    print("Acc:", np.mean(accuracies))

    # ModelGenome

    layers_to_model_genome = lambda layers: SeqEvoModelGenome.create_with_default_params(SeqEvoGenome(layers=layers))
    fitness = Fitness([], [], [], [], X_y_validation_splits, None).small_split_kfold_acc
    model_genome = layers_to_model_genome(model_genome)
    print("Fitness with fitness func:", fitness(model_genome, log_func=print))


    

