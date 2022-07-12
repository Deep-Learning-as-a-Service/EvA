# pylint: disable=locally-disabled, multiple-statements, fixme, line-too-long, no-name-in-module, unused_import, wrong-import-order, bad-option-value
# some imports are not accepted by pylint

from models.RainbowModel import RainbowModel
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import h5py
from tensorflow.keras import regularizers
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
)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam 
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import itertools

from datetime import datetime
import os
from utils.typing import assert_type
from utils.Window import Window
from utils.Recording import Recording
import itertools


class JensModel(RainbowModel):
    def __init__(self, **kwargs):
        """
        :param kwargs:
            window_size: int
            n_features: int
            n_outputs: int
        
        - leave recording out
        - higher level opportunity
        - 3 sec window (90)
        -> accuracy: 0.63
        """

        self.model_name = "jens_model"
        super().__init__(**kwargs)

    def _create_model(self):

        i = Input(shape=(self.window_size, self.n_features, 1))  # before: self.x_train[0].shape - (25, 51, 1)... before self_x_train = np.expand_dims(self.x_train[0], -1) - around the value another []
        if self.add_preprocessing_layer:
            x = self._preprocessing_layer(i)
        else:
            x = i
        x = Conv2D(
            32,
            (3, 3),
            strides=2,
            activation="relu",
            padding="same",
            kernel_regularizer=regularizers.l2(0.0005),
        )(x)
        x = BatchNormalization()(x)
        x = MaxPooling2D((2, 2))(x)
        x = Dropout(0.2)(x)

        x = Conv2D(
            64,
            (3, 3),
            strides=2,
            activation="relu",
            padding="same",
            kernel_regularizer=regularizers.l2(0.0005),
        )(x)
        x = BatchNormalization()(x)
        x = Dropout(0.4)(x)

        x = Conv2D(
            128,
            (3, 3),
            strides=2,
            activation="relu",
            padding="same",
            kernel_regularizer=regularizers.l2(0.0005),
        )(x)
        x = BatchNormalization()(x)
        x = MaxPooling2D((2, 2))(x)
        x = Dropout(0.2)(x)

        x = Flatten()(x)
        x = Dropout(0.2)(x)
        x = Dense(1024, activation="relu")(x)
        x = Dropout(0.2)(x)
        x = Dense(self.n_outputs, activation="softmax")(x)
        model = Model(i, x)
        model.compile(
            optimizer=Adam(learning_rate=self.learning_rate),
            loss="CategoricalCrossentropy", # CategoricalCrossentropy (than we have to to the one hot encoding - to_categorical), before: "sparse_categorical_crossentropy"
            metrics=["accuracy"],
        )

        return model

