import os
import numpy as np
from datetime import datetime
from evaluation.metrics import accuracy
from loader.get_data import get_data
from loader.load_dataset import load_dataset
from loader.load_lab_dataset import load_lab_dataset
from model_representation.ModelGenome.SeqEvoModelGenome import SeqEvoModelGenome
from model_representation.ParametrizedLayer.ParametrizedLayer import ParametrizedLayer
from models.CNNLstm import CNNLstm
from models.InnoHAR import InnoHAR
from optimizer.HyPaOptuna.ModelOptuna2 import ModelOptuna2
from optimizer.SeqEvo.SeqEvoHistory import SeqEvoHistory
import utils.settings as settings
from model_representation.ParametrizedLayer.PDenseLayer import PDenseLayer
from model_representation.ParametrizedLayer.PConv2DLayer import PConv2DLayer
from model_representation.ParametrizedLayer.PLstmLayer import PLstmLayer
from utils.logger import logger as log_func
from models.ResNetModel import ResNetModel
from keras import backend as K
import utils.config as config

testing = False
optuna_iterations = 20


experiment_name = "opp_evaluation_bestmodel_vs_others"

config.telegram_chat_id = "-1001707088052"

currentDT = datetime.now()
currentDT_str = currentDT.strftime("%y-%m-%d_%H-%M-%S_%f")
experiment_name = experiment_name + "-" + currentDT_str
logger = lambda *args, **kwargs: log_func(*args, path=f"logs/{experiment_name}", **kwargs) if not testing else print(*args, **kwargs)
prio_logger = lambda *args, **kwargs: logger(*args, prio=True, **kwargs) if not testing else print(*args, **kwargs)

prio_logger(f"starting {experiment_name}")

# Config --------------------------------------------------------------------------
window_size = 90
n_features = 51
n_classes = 6
num_folds = 5

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

# models
hist_genomes = [hist_genome for hist_genome in SeqEvoHistory(
    path_to_file=f'data/opportunity/seqevo_history.csv'
).read()]
best_hist_genome = [hist_genome for hist_genome in hist_genomes if hist_genome.fitness == max([hist_genome.fitness for hist_genome in hist_genomes])][0]

prio_logger( "========================= Optuna Optimized Hyperparams ========================")

models = []

# our best performer
models.append(SeqEvoModelGenome.create_with_default_params(best_hist_genome.seqevo_genome))

hyperparams = []

for model in models:
    prio_logger(f"====================={model.__class__.__name__}==========================")
    is_seqevo = model.__class__.__name__ =='SeqEvoModelGenome'

    optuna_params = ModelOptuna2(
        model = model,
        n_trials=optuna_iterations,
        X_y_val_splits=X_y_validation_splits,
        log_func=prio_logger,
        is_seqevo=is_seqevo
    ).run()
    n_epochs = optuna_params["n_epochs"]
    batch_size = optuna_params["batch_size"]
    learning_rate = optuna_params["learning_rate"]
    hyperparams.append((n_epochs, batch_size, learning_rate))
    prio_logger(optuna_params.items())

