import os
import numpy as np
from datetime import datetime
from evaluation.metrics import accuracy
from loader.get_data import get_data
from loader.load_dataset import load_dataset
from model_representation.ParametrizedLayer.ParametrizedLayer import ParametrizedLayer
import utils.settings as settings

from utils.folder_operations import new_saved_experiment_folder
from model_representation.ParametrizedLayer.PDenseLayer import PDenseLayer
from model_representation.ParametrizedLayer.PConv1DLayer import PConv1DLayer
from model_representation.ParametrizedLayer.PConv2DLayer import PConv2DLayer
from model_representation.ParametrizedLayer.PLstmLayer import PLstmLayer
from utils.logger import logger as log_func
from loader.load_lab_dataset import load_lab_dataset

from models.ResNetModel import ResNetModel

def _model_fit_test(model, X_train_fit, y_train_fit, X_test_fit, y_test_fit, eval_func):
        model.fit(
            X_train_fit, 
            y_train_fit
        )
        model.model.summary()
        y_test_pred = model.predict(X_test_fit)
        fitness = eval_func(y_test_fit, y_test_pred)
        return fitness

# Experiment Name ---------------------------------------------------------------
experiment_name = "resnet_rainbow_lab"
currentDT = datetime.now()
currentDT_str = currentDT.strftime("%y-%m-%d_%H-%M-%S_%f")
experiment_name = experiment_name + "-" + currentDT_str
logger = lambda *args, **kwargs: log_func(*args, path=f"logs/{experiment_name}", **kwargs)
prio_logger = lambda *args, **kwargs: logger(*args, prio=True, **kwargs)

prio_logger(f"starting {experiment_name}")


# Config --------------------------------------------------------------------------

features = ["dq_W_LF", "dq_X_LF", "dq_Y_LF", "dq_Z_LF", "dv[1]_LF", "dv[2]_LF", "dv[3]_LF", "Mag_X_LF", "Mag_Y_LF", "Mag_Z_LF", "dq_W_LW", "dq_X_LW", "dq_Y_LW", "dq_Z_LW", "dv[1]_LW", "dv[2]_LW", "dv[3]_LW", "Mag_X_LW", "Mag_Y_LW", "Mag_Z_LW", "dq_W_ST", "dq_X_ST", "dq_Y_ST", "dq_Z_ST", "dv[1]_ST",
            "dv[2]_ST", "dv[3]_ST", "Mag_X_ST", "Mag_Y_ST", "Mag_Z_ST", "dq_W_RW", "dq_X_RW", "dq_Y_RW", "dq_Z_RW", "dv[1]_RW", "dv[2]_RW", "dv[3]_RW", "Mag_X_RW", "Mag_Y_RW", "Mag_Z_RW", "dq_W_RF", "dq_X_RF", "dq_Y_RF", "dq_Z_RF", "dv[1]_RF", "dv[2]_RF", "dv[3]_RF", "Mag_X_RF", "Mag_Y_RF", "Mag_Z_RF"]
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

window_size = 900 # 15 seconds (60 Hz * 15 s)
n_features = len(features)
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
    activityLabelToIndexMap=category_labels,
    features=features
)
X_y_validation_splits = get_data(
    load_recordings=load_lab_data,
    shuffle_seed=1678978086101,
    num_folds=num_folds
)




# models
model_classes = [ResNetModel]

for model_class in model_classes:
    model_name = model_class.__name__
    
    # or JensModel
    

    fitnesses = []
    idx = 0
    for X_train_split, y_train_split, X_val_split, y_val_split in X_y_validation_splits:
        idx += 1
        model = model_class(
            window_size=window_size, 
            n_features=n_features, 
            n_outputs=n_classes, 
            n_epochs=10, 
            learning_rate=0.0001, 
            batch_size=32,
            add_preprocessing_layer=True
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

    # Create Folder, save model export and evaluations there
    experiment_folder_path = new_saved_experiment_folder(experiment_name) # create folder to store results
    

