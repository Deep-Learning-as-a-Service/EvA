from model_representation.ParametrizedLayer.ParametrizedLayer import ParametrizedLayer
import random
from tensorflow import keras
from model_representation.EvoParam.IntEvoParam import IntEvoParam
from model_representation.EvoParam.FloatEvoParam import FloatEvoParam
from model_representation.EvoParam.CategEvoParam import CategEvoParam
from keras.layers import Dropout

# TODO: changed value range
class DenseUnitsParam(IntEvoParam):
    _default_values = [32, 64, 128, 256, 512]
    _value_range = [16, 1024]
    _key = "units"
    _mean = 514
    _sd = 2000

class DenseDropoutParam(CategEvoParam):
    """
    after layer param!
    - will not get passed in the _layer function from the PLayer
    - will create its own layer, therefore has a _layer() method
    """
    _default_values = [0.0, 0.1]
    _value_range = [0.0, 0.1, 0.15, 0.2, 0.25, 0.3]
    _weights = [0.5, 0.1, 0.1, 0.1, 0.1, 0.1]
    _key = "dropout"

    def _layer(self):
        def dropout(tensor):
            if self._value != 0:
                return Dropout(rate=self._value)(tensor)
            return tensor
        return dropout

class PDenseLayer(ParametrizedLayer):
    _layer = lambda **kwargs: keras.layers.Dense(activation="relu", **kwargs)
    _param_classes = [DenseUnitsParam, DenseDropoutParam]
    _after_layer_params = [DenseDropoutParam]
            