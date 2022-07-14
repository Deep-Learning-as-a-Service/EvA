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
from models.ResNetModel import ResNetModel
from models.LeanderDeepConvLSTM import LeanderDeepConvLSTM

experiment_name = "dl_project"
currentDT = datetime.now()
currentDT_str = currentDT.strftime("%y-%m-%d_%H-%M-%S_%f")
experiment_name = experiment_name + "-" + currentDT_str
logger = lambda *args, **kwargs: log_func(*args, path=f"logs/{experiment_name}", **kwargs)
# prio_logger = lambda *args, **kwargs: logger(*args, prio=True, **kwargs)

# we will do kfold - this is how we get the fitness for one split
def _model_fit_test(model, X_train_fit, y_train_fit, X_test_fit, y_test_fit, eval_func):
    model.fit(
        X_train_fit, 
        y_train_fit
    )
    y_test_pred = model.predict(X_test_fit)
    fitness = eval_func(y_test_fit, y_test_pred)
    return fitness

logger(f"starting {experiment_name}")

# Config
window_size = 30*5 # 5 seconds windows (30 Hz)
n_features = 51
n_classes = 6
num_folds = 2 # I normally use 5 folds (5 times training, 5 times evaluation), with 10 epochs this takes a while, so I use a gpupro from the dhclab, if you run it locally I suggest take less folds

# layer_pool is needed for evolution, ignore it for now
layer_pool: 'list[ParametrizedLayer]' = [PConv2DLayer, PDenseLayer, PLstmLayer] #PConv1DLayer
data_dimension_dict = {
    "window_size": window_size,
    "n_features": n_features,
    "n_classes": n_classes
}
# this is basically a way to have global vars in python
settings.init(_layer_pool=layer_pool, _data_dimension_dict=data_dimension_dict)

load_recs = lambda: load_dataset(os.path.join(settings.opportunity_dataset_csv_path, 'data.csv'), 
    label_column_name='ACTIVITY_IDX', 
    recording_idx_name='RECORDING_IDX', 
    column_names_to_ignore=['SUBJECT_IDX', 'MILLISECONDS']
)

# load, preprocess and kfold split data
X_y_validation_splits = get_data(
    load_recordings=load_recs,
    shuffle_seed=1678978086101,
    num_folds=num_folds
)
# X_y_validation_splits: [(X_train_01, y_train_01, X_val_01, y_val_01), (X_train_02, y_train_02, X_val_02, y_val_02), ...]


# evaluate multiple models - for now we use a single model
model_classes = [LeanderDeepConvLSTM]

for model_class in model_classes:
    model_name = model_class.__name__

    fitnesses = []
    idx = 0
    for X_train_split, y_train_split, X_val_split, y_val_split in X_y_validation_splits:
        idx += 1
        model = model_class(
            window_size=window_size, 
            n_features=n_features, 
            n_outputs=n_classes, 
            n_epochs=10, 
            learning_rate=0.001, 
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

        # these are the important logs!!
        logger(f"fitness on {idx}. fold: {fitness}")
    logger(f"average fitness of all splits: {np.mean(fitnesses)}")

    # if you want create a saved_experiments folder and store a text file with the resulsts
    # experiment_folder_path = new_saved_experiment_folder(experiment_name)
    

