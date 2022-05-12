from model_representation.ParametrizedLayer.ParametrizedLayer import ParametrizedLayer
from model_representation.EvoParam.IntEvoParam import IntEvoParam
from tensorflow import keras

class LstmUnitsParam(IntEvoParam):
    _default_values = [4, 8, 16]
    _value_range = [2, 32]
    _key = "units"


class PLstmLayer(ParametrizedLayer):
    _layer = lambda **kwargs: keras.layers.LSTM(return_sequences=True, **kwargs)
    _param_classes = [LstmUnitsParam]