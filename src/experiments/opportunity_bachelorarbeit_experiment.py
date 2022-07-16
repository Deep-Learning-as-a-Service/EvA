import os
import numpy as np
from datetime import datetime
from evaluation.metrics import accuracy
from loader.get_data import get_data
from loader.load_dataset import load_dataset
from model_representation.ModelGenome.SeqEvoModelGenome import SeqEvoModelGenome
from model_representation.ParametrizedLayer.ParametrizedLayer import ParametrizedLayer
from models.CNNLstm import CNNLstm
from models.DeepConvLSTM import DeepConvLSTM
from models.InnoHAR import InnoHAR
from models.LeanderDeepConvLSTM import LeanderDeepConvLSTM
from optimizer.HyPaOptuna.ModelOptuna import ModelOptuna
from optimizer.SeqEvo.SeqEvoHistory import SeqEvoHistory
import utils.settings as settings

from utils.folder_operations import new_saved_experiment_folder
from model_representation.ParametrizedLayer.PDenseLayer import PDenseLayer
from model_representation.ParametrizedLayer.PConv1DLayer import PConv1DLayer
from model_representation.ParametrizedLayer.PConv2DLayer import PConv2DLayer
from model_representation.ParametrizedLayer.PLstmLayer import PLstmLayer
from models.ShallowDeepConvLSTM import ShallowDeepConvLSTM
from utils.logger import logger as log_func
from models.ResNetModel import ResNetModel
from keras import backend as K
import utils.config as config

def _model_fit_test(model, X_train_fit, y_train_fit, X_test_fit, y_test_fit):
    model.fit(
        X_train_fit, 
        y_train_fit
    )
    y_test_pred = model.predict(X_test_fit)
    fitness = accuracy(y_test_fit, y_test_pred)
    return fitness

def _model_fit_test_after_optuna(model, n_epochs, n_batch_size, lr, X_train_fit, y_train_fit, X_test_fit, y_test_fit):

        # set learning rate
        K.set_value(model.optimizer.learning_rate, lr)

        model.fit(
            X_train_fit, 
            y_train_fit,
            epochs=n_epochs,
            batch_size=n_batch_size,
        )

        y_test_pred = model.predict(X_test_fit)

        fitness = accuracy(y_test_fit, y_test_pred)
        return fitness

experiment_name = "opportunity_evaluation_bestmodel_vs_others"

config.telegram_chat_id = "-1001555874641"

currentDT = datetime.now()
currentDT_str = currentDT.strftime("%y-%m-%d_%H-%M-%S_%f")
experiment_name = experiment_name + "-" + currentDT_str
logger = lambda *args, **kwargs: log_func(*args, path=f"logs/{experiment_name}", **kwargs)
prio_logger = lambda *args, **kwargs: logger(*args, prio=True, **kwargs)

prio_logger(f"starting {experiment_name}")

# Data
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
model_classes = [CNNLstm, ResNetModel, InnoHAR]
hist_genomes = [hist_genome for hist_genome in SeqEvoHistory(
    path_to_file=f'data/opportunity/seqevo_history.csv'
).read()]
best_hist_genome = [hist_genome for hist_genome in hist_genomes if hist_genome.fitness == max([hist_genome.fitness for hist_genome in hist_genomes])][0]


prio_logger( "========================= Default Trainings Params ========================")

for model_class in model_classes:
    model_name = model_class.__name__
    
    logger("========================================================")
    prio_logger(f"====================={model_name}==========================")
    logger("========================================================")

    fitnesses = []
    idx = 0
    for X_train_split, y_train_split, X_val_split, y_val_split in X_y_validation_splits:
        idx += 1
        model = model_class(
            window_size=window_size, 
            n_features=n_features, 
            n_outputs=n_classes, 
            n_epochs=5, 
            learning_rate=0.001, 
            batch_size=32,
            add_preprocessing_layer=True,
        )

        fitness = _model_fit_test(model=model, 
            X_train_fit=X_train_split, 
            y_train_fit=y_train_split, 
            X_test_fit=X_val_split, 
            y_test_fit=y_val_split, 
            eval_func=accuracy
        )
        fitnesses.append(fitness)
        prio_logger(f"fitness on {idx}. fold: {fitness}")
    prio_logger(f"average fitness of all splits: {np.mean(fitnesses)}")

logger("========================================================")
prio_logger(f"===================== UNSER BESTPERFORMER ==========================")
logger("========================================================")


fitnesses = []
idx = 0
for X_train_split, y_train_split, X_val_split, y_val_split in X_y_validation_splits:  
    idx += 1
    best_keras_model = SeqEvoModelGenome.create_with_default_params(best_hist_genome.seqevo_genome).get_model()
    fitness = _model_fit_test(model = best_keras_model,
            X_train_fit=X_train_split, 
            y_train_fit=y_train_split, 
            X_test_fit=X_val_split, 
            y_test_fit=y_val_split
            )
    fitnesses.append(fitness)
    prio_logger(f"fitness on {idx}. fold: {fitness}")
prio_logger(f"average fitness of all splits: {np.mean(fitnesses)}")

prio_logger( "========================= Optuna Optimized Hyperparams ========================")

models = []
# our best performer
models.append(SeqEvoModelGenome.create_with_default_params(best_hist_genome.seqevo_genome).get_model())

# all others
for model_class in model_classes:
    models.append(model_class(
            window_size=window_size, 
            n_features=n_features, 
            n_outputs=n_classes, 
            n_epochs=5, 
            learning_rate=0.001, 
            batch_size=32,
            add_preprocessing_layer=True
        ))

# deepcopy of all model instances, to 
hyperparams = []

for model in models:
    prio_logger(f"====================={model.__class__.__name__}==========================")
    optuna_params = ModelOptuna(
        model = model,
        n_trials=100,
        X_train_fit=X_y_validation_splits[0][0],
        y_train_fit=X_y_validation_splits[0][1],
        X_test_fit=X_y_validation_splits[0][2],
        y_test_fit=X_y_validation_splits[0][3],
        log_func=logger
    ).run()
    n_epochs = optuna_params["n_epochs"]
    batch_size = optuna_params["batch_size"]
    learning_rate = optuna_params["learning_rate"]
    hyperparams.append((n_epochs, batch_size, learning_rate))
    prio_logger(optuna_params.items())


#########################################################################################
# new compiled instances of the models, to avoid memory of the weights (dirty approach) #
#########################################################################################

models_for_validation = []
models_for_validation.append(SeqEvoModelGenome.create_with_default_params(best_hist_genome.seqevo_genome).get_model())

# (hyperparams will get overwritten later, don't worry, once again ugly but should work)
for model_class in model_classes:
    models_for_validation.append(model_class(
            window_size=window_size, 
            n_features=n_features, 
            n_outputs=n_classes, 
            n_epochs=5, 
            learning_rate=0.001, 
            batch_size=32,
            add_preprocessing_layer=True
        ))

for i, model in enumerate(models_for_validation):
    model_name = model.__class__.__name__
    
    logger("========================================================")
    prio_logger(f"====================={model_name}==========================")
    logger("========================================================")

    fitnesses = []
    idx = 0
    for X_train_split, y_train_split, X_val_split, y_val_split in X_y_validation_splits:
        idx += 1

        fitness = _model_fit_test_after_optuna(model=model,
            n_epochs = hyperparams[i][0],
            n_batch_size= hyperparams[i][1],
            lr = hyperparams[i][2],
            X_train_fit=X_train_split, 
            y_train_fit=y_train_split, 
            X_test_fit=X_val_split, 
            y_test_fit=y_val_split
            )
        fitnesses.append(fitness)
        prio_logger(f"fitness on {idx}. fold: {fitness}")
    prio_logger(f"average fitness of all splits: {np.mean(fitnesses)}")



