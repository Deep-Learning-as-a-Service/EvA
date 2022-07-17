import keras
from keras import regularizers
from keras.models import Sequential
from keras.layers import MaxPooling1D, Conv1D, Flatten, Dense, Concatenate, GRU, Input, InputLayer
from models.RainbowModel import RainbowModel
import utils.settings as settings
import tensorflow as tf

# implementation mostly from https://github.com/rht6226/InnoHAR-NeuralNet/blob/master/InnoHAR_Model.ipynb

class Incept1D(keras.Model):
    def __init__(self, c1, c2, c3, c4):
        super(Incept1D, self).__init__()
        self.p1_1x1_conv = Conv1D(filters=c1, kernel_size=1, activation='relu', padding='same')

        self.p2_1x1_conv = Conv1D(filters=c2[0], kernel_size=1, activation='relu', padding='same')
        self.p2_1x3_conv = Conv1D(filters=c2[1], kernel_size=3, activation='relu', padding='same')

        self.p3_1x1_conv = Conv1D(filters=c3[0], kernel_size=1, activation='relu', padding='same')
        self.p3_1x5_conv = Conv1D(filters=c3[1], kernel_size=3, activation='relu', padding='same')

        self.p4_1x3_maxpool = MaxPooling1D(pool_size=3, strides=1, padding='same')
        self.p4_1x1_conv = Conv1D(filters=c4, kernel_size=1, padding='same', activation='relu')


    def call(self, x):
        p1 = self.p1_1x1_conv(x)
        p2 = self.p2_1x3_conv(self.p2_1x1_conv(x))
        p3 = self.p3_1x5_conv(self.p3_1x1_conv(x))
        p4 = self.p4_1x1_conv(self.p4_1x3_maxpool(x))
        # Concatenate the outputs on the channel dimension
        return Concatenate()([p1, p2, p3, p4])
class InnoHAR(RainbowModel):
    

    def _create_model(self):

        model = Sequential(name='InnoHAR')
        model.add(tf.keras.layers.Input((settings.data_dimension_dict["window_size"], settings.data_dimension_dict["n_features"])))
        if self.add_preprocessing_layer:
            model.add(self._preprocessing_layer())

        model.add(Incept1D(64, (32, 64), (32, 64), 64))
        model.add(Incept1D(56, (32, 64), (32, 64), 64))
        model.add(Incept1D(32, (32, 56), (32, 56), 32))
        model.add(MaxPooling1D(pool_size=2, padding='same'))
        model.add(Incept1D(32, (16, 32), (16, 32), 24))
        model.add(MaxPooling1D(pool_size=2, padding='same'))
        model.add(GRU(120, return_sequences=True))
        model.add(GRU(40))
        model.add(Dense(settings.data_dimension_dict["n_classes"], activation='softmax',
                        kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4), 
                        bias_regularizer=regularizers.l2(1e-4), 
                        activity_regularizer=regularizers.l2(1e-5)
                        ))

        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

        return model
    def _preprocessing_layer(self) -> tf.keras.layers.Layer:
        return tf.keras.layers.Normalization(
            axis=-1,
            variance=settings.input_distribution_variance,
            mean=settings.input_distribution_mean,
        )