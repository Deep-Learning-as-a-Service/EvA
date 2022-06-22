from model_representation.ParametrizedLayer.ParametrizedLayer import ParametrizedLayer
from model_representation.EvoParam.IntEvoParam import IntEvoParam
from model_representation.EvoParam.FloatEvoParam import FloatEvoParam
from tensorflow.keras.initializers import Orthogonal
from tensorflow import keras

# TODO: changed value range
class LstmUnitsParam(IntEvoParam):
    _default_values = [4, 8, 16]
    _value_range = [2, 1028]
    _key = "units"
    _mean = 514
    _sd = 400

class LstmDropoutParam(FloatEvoParam):
    _default_values = [0.1, 0.2]
    _value_range = [0.0, 0.4]
    _key = "dropout"
    _mean = 0.15
    _sd = 0.05

class PLstmLayer(ParametrizedLayer):
    _layer = lambda **kwargs: keras.layers.LSTM(return_sequences=True, kernel_initializer=Orthogonal(), **kwargs)
    _param_classes = [LstmUnitsParam, LstmDropoutParam]