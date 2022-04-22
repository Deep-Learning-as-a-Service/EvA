import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from models.JensModel import JensModel
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

class SimpleModel(JensModel):

    def _create_model(self, n_features, n_outputs, window_size):

        input_net = Input(shape=(window_size, n_features, 1))
  
        ## Encoder starts
        conv1 = Conv2D(32, 3, strides=(2,2), activation = 'relu', padding = 'same')(input_net)
        conv2 = Conv2D(64, 3, strides=(2,2), activation = 'relu', padding = 'same')(conv1)
        conv3 = Conv2D(128, 3, strides=(2,2), activation = 'relu', padding = 'same')(conv2)

        conv4 = Conv2D(128, 3, strides=(2,2), activation = 'relu', padding = 'same')(conv3)

        ## And now the decoder
        up1 = Conv2D(128, 3, activation = 'relu', padding = 'same')(UpSampling2D(size = (2,2))(conv4))
        merge1 = concatenate([conv3,up1], axis = 3)
        up2 = Conv2D(64, 3, activation = 'relu', padding = 'same')(UpSampling2D(size = (2,2))(merge1))
        merge2 = concatenate([conv2,up2], axis = 3)
        up3 = Conv2D(32, 3, activation = 'relu', padding = 'same')(UpSampling2D(size = (2,2))(merge2))
        merge3 = concatenate([conv1,up3], axis = 3)

        up4 = Conv2D(32, 3, padding = 'same')(UpSampling2D(size = (2,2))(merge3))

        output_net = Conv2D(3, 3, padding = 'same')(up4)

        # added
        output_net = Dense(n_outputs, activation="softmax")(output_net)

        model = Model(inputs = input_net, outputs = output_net)

        model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
        model.summary()
        return model