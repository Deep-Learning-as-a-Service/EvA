from model_representation.ParametrizedLayer.ParametrizedLayer import ParametrizedLayer
from model_representation.EvoParam.IntEvoParam import IntEvoParam
from tensorflow import keras

# TODO: changed value range
class LstmUnitsParam(IntEvoParam):
    _default_values = [4, 8, 16]
    _value_range = [2, 1028]
    _key = "units"
    _mean = 8
    _sd = 4

class PLstmLayer(ParametrizedLayer):
    _layer = lambda **kwargs: keras.layers.LSTM(return_sequences=True, **kwargs)
    _param_classes = [LstmUnitsParam]