"""
State-of-the-art, Standard models, what is currently possible on the dataset
"""


import os
import random
import numpy as np
import pandas as pd
from datetime import datetime
from loader.load_dataset import load_dataset
from loader.get_opportunity_data import get_opportunity_data
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
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from models.RainbowModel import RainbowModel
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam 
from tensorflow.keras.initializers import Orthogonal
from tensorflow.keras.layers import (
    Input,
    Conv2D,
    Dense,
    Flatten,
    Dropout,
    LSTM,
    GlobalMaxPooling1D,
    MaxPooling2D,
    BatchNormalization,
    concatenate,
    Reshape,
    Permute,
    LSTM
)
import optuna

# Preprocessing ------

layer_pool: 'list[ParametrizedLayer]' = [PConv2DLayer, PDenseLayer, PLstmLayer] #PConv1DLayer
settings.init(_layer_pool=layer_pool)
experiment_name = "competing_architectures"

currentDT = datetime.now()
currentDT_str = currentDT.strftime("%y-%m-%d_%H-%M-%S_%f")
experiment_name = experiment_name + "-" + currentDT_str

# Data
window_size = 30 * 3
n_classes = 6
n_outputs = n_classes
n_features = 51
X_train, y_train, X_test, y_test, X_y_validation_splits = get_opportunity_data(
    window_size=window_size,
    n_features=n_features,
    n_classes=n_classes 
)

def create_leander_deep_conv_lstm(conv1_n_filters, conv2_n_filters, lstm_n_units):

    initializer = Orthogonal()
    conv_layer = lambda n_filters: lambda the_input: Conv2D(filters=n_filters, strides=(5, 1), kernel_size=(5, 1), activation="relu", kernel_initializer=initializer)(the_input)
    lstm_layer = lambda the_input: LSTM(units=lstm_n_units, dropout=0.1, return_sequences=True, kernel_initializer=initializer)(the_input) # 32 units default

    i = Input(shape=(window_size, n_features))

    # Adding 4 CNN layers.
    x = Reshape(target_shape=(window_size, n_features, 1))(i)
    conv_n_filters = [conv1_n_filters, conv2_n_filters] # 32, 64
    for n_filters in conv_n_filters:
        x = conv_layer(n_filters=n_filters)(x)

    x = Reshape(
        (
            int(x.shape[1]),
            int(x.shape[2]) * int(x.shape[3]),
        )
    )(x)

    for _ in range(1):
        x = lstm_layer(x)

    x = Flatten()(x)
    x = Dense(units=n_outputs, activation="softmax")(x)

    model = Model(i, x)
    model.compile(
        optimizer="RMSprop", # keras.optimizers.Adam(learning_rate=0.01)
        loss="CategoricalCrossentropy",  # CategoricalCrossentropy (than we have to to the one hot encoding - to_categorical), before: "sparse_categorical_crossentropy"
        metrics=["accuracy"],
    )

    return model

def get_fitness(model, n_epochs, batch_size):


    history = model.fit(X_train, y_train, epochs=n_epochs, batch_size=batch_size, verbose=1)

    y_test_pred = model.predict(X_test)
    y_test_pred_numbers = np.argmax(y_test_pred, axis=1)
    y_test_numbers = np.argmax(y_test, axis=1)

    # Create Folder, save model export and evaluations there
    # experiment_folder_path = new_saved_experiment_folder(experiment_name) # create folder to store results

    # model.export(experiment_folder_path) # opt: export model to folder
    # create_conf_matrix(experiment_folder_path, y_test_pred, y_test, model_name)
    f1 = f1_score(y_true=y_test_numbers, y_pred=y_test_pred_numbers, average="weighted")
    acc = np.sum(y_test_pred_numbers == y_test_numbers) / len(y_test_numbers)

    # text metrics
    # with open(os.path.join(experiment_folder_path, f'{model_name}.txt'), "w") as f:
    #     f.write(f"Accuracy: {acc}\n")
    #     f.write(f"F1-Score: {f1}")
    return acc, history
    


# Define an objective function to be minimized.
def objective(trial):

    # Hyperparam Architecture
    conv1_n_filters = 32 + trial.suggest_int("conv1_n_filters", 32, 128) # 32
    conv2_n_filters = 64 # trial.suggest_int("conv2_n_filters", 32, 128)
    lstm_n_units = 32 # trial.suggest_int("lstm_n_units", 32, 128)
    model = create_leander_deep_conv_lstm(conv1_n_filters, conv2_n_filters, lstm_n_units)

    # Hyperparam Training
    n_epochs = trial.suggest_int("n_epochs", 1, 10)
    batch_size = trial.suggest_int("batch_size", 1, 32)
    acc, history = get_fitness(model, n_epochs, batch_size)

    return 1 - acc


# Main ------
study = optuna.create_study()  # Create a new study.
study.optimize(objective, n_trials=3)  # Invoke optimization of the objective function.
print(study.best_params)