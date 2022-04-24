from tensorflow import keras
from tensorflow.keras import layers
from models.JensModel import JensModel


class SimpleModel(JensModel):
    """
    https://keras.io/examples/vision/mnist_convnet/
    """

    def _create_model(self, n_features, n_outputs, window_size):

        model = keras.Sequential(
            [
                keras.Input(shape=(window_size, n_features, 1)),
                layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
                layers.MaxPooling2D(pool_size=(2, 2)),
                layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
                layers.MaxPooling2D(pool_size=(2, 2)),
                layers.Flatten(),
                layers.Dropout(0.5),
                layers.Dense(n_outputs, activation="softmax"),
            ]
        )

        model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
        model.summary()
        return model