import keras
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, TimeDistributed, Conv1D, MaxPooling1D, Flatten, InputLayer

from models.RainbowModel import RainbowModel
import utils.settings as settings
import tensorflow as tf

class CNNLstm(RainbowModel):

    def _create_model(self): 
        model = tf.keras.models.Sequential([
            tf.keras.layers.Input((settings.data_dimension_dict["window_size"], settings.data_dimension_dict["n_features"])),
            self._preprocessing_layer(),
            tf.keras.layers.Conv1D(filters=64, kernel_size=5, strides=1, activation="relu"),
            tf.keras.layers.MaxPooling1D(pool_size=2, strides=1, padding="valid"),
            tf.keras.layers.Conv1D(filters=32, kernel_size=3, strides=1, activation="relu"),
            tf.keras.layers.MaxPooling1D(pool_size=2, strides=1, padding="valid"),
            tf.keras.layers.LSTM(128, return_sequences=True),
            tf.keras.layers.LSTM(128, return_sequences=True),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(16, activation="relu"),
            tf.keras.layers.Dropout(0.1),
            tf.keras.layers.Dense(settings.data_dimension_dict["n_classes"], activation='softmax')
        ])

        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

        return model

    def _preprocessing_layer(self) -> tf.keras.layers.Layer:
        return tf.keras.layers.Normalization(
            axis=-1,
            variance=settings.input_distribution_variance,
            mean=settings.input_distribution_mean,
        )