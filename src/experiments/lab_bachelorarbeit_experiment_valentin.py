import numpy as np
from datetime import datetime
from evaluation.metrics import accuracy
from loader.get_data import get_data
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

def _model_fit_test(model, X_train_fit, y_train_fit, X_test_fit, y_test_fit):
    model.fit(
        X_train_fit, 
        y_train_fit
        )
    y_test_pred = model.predict(X_test_fit)
    fitness = accuracy(y_test_fit, y_test_pred)
    return fitness

def _model_fit_test_after_optuna(model, n_epochs, n_batch_size, lr, X_train_fit, y_train_fit, X_test_fit, y_test_fit, is_seqevo):

        keras_model = None
        if is_seqevo:
            keras_model = model.get_model()
        else:
            keras_model = model._create_model()

        # set learning rate
        K.set_value(keras_model.optimizer.learning_rate, lr)

        keras_model.fit(
            X_train_fit, 
            y_train_fit,
            epochs=n_epochs,
            batch_size=n_batch_size,
        )

        y_test_pred = keras_model.predict(X_test_fit)

        fitness = accuracy(y_test_fit, y_test_pred)
        return fitness

experiment_name = "lab_evaluation_bestmodel_vs_others"

config.telegram_chat_id = "-1001731938222"

currentDT = datetime.now()
currentDT_str = currentDT.strftime("%y-%m-%d_%H-%M-%S_%f")
experiment_name = experiment_name + "-" + currentDT_str
logger = lambda *args, **kwargs: log_func(*args, path=f"logs/{experiment_name}", **kwargs) if not testing else print(*args, **kwargs)
prio_logger = lambda *args, **kwargs: logger(*args, prio=True, **kwargs) if not testing else print(*args, **kwargs)

prio_logger(f"starting {experiment_name}")

# Config --------------------------------------------------------------------------
category_labels = {
    "null - activity": 0,
    "aufräumen": 1,
    "aufwischen (staub)": 2,
    "bett machen": 3,
    "dokumentation": 4,
    "umkleiden": 5,
    "essen reichen": 6,
    "gesamtwaschen im bett": 7,
    "getränke ausschenken": 8,
    "haare kämmen": 9,
    "waschen am waschbecken": 10,
    "medikamente stellen": 11,
    "rollstuhl schieben": 12,
    "rollstuhl transfer": 13
}

window_size = 30*3
n_features = 70
n_classes = len(category_labels)
num_folds = 5
validation_iterations = 3

layer_pool: 'list[ParametrizedLayer]' = [PConv2DLayer, PDenseLayer, PLstmLayer] #PConv1DLayer
data_dimension_dict = {
    "window_size": window_size,
    "n_features": n_features,
    "n_classes": n_classes
}
settings.init(_layer_pool=layer_pool, _data_dimension_dict=data_dimension_dict)

load_lab_data = lambda: load_lab_dataset(
    path="../../data/lab_data_filtered_without_null", 
    activityLabelToIndexMap=category_labels
)
X_y_validation_splits = get_data(
    load_recordings=load_lab_data,
    shuffle_seed=1678978086101,
    num_folds=num_folds
)

# models
model_classes = [CNNLstm, ResNetModel, InnoHAR]
hist_genomes = [hist_genome for hist_genome in SeqEvoHistory(
    path_to_file=f'data/lab/seqevo_history.csv'
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
            add_preprocessing_layer=True
        )._create_model()

        fitness = _model_fit_test(model=model, 
            X_train_fit=X_train_split, 
            y_train_fit=y_train_split, 
            X_test_fit=X_val_split, 
            y_test_fit=y_val_split
            ) if not testing else 0.1
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
            ) if not testing else 0.1
    fitnesses.append(fitness)
    prio_logger(f"fitness on {idx}. fold: {fitness}")
prio_logger(f"average fitness of all splits: {np.mean(fitnesses)}")

prio_logger( "========================= Optuna Optimized Hyperparams ========================")

models = []

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
        log_func=logger,
        is_seqevo=is_seqevo
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

models_for_validation.append(SeqEvoModelGenome.create_with_default_params(best_hist_genome.seqevo_genome))


for i, model in enumerate(models_for_validation):
    is_seqevo = model.__class__.__name__ =='SeqEvoModelGenome'

    
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
            y_test_fit=y_val_split,
            is_seqevo=is_seqevo
            ) if not testing else 0.1


        fitnesses.append(fitness)
        prio_logger(f"fitness on {idx}. fold: {fitness}")
    prio_logger(f"average fitness of all splits: {np.mean(fitnesses)}")



