from model_representation.ParametrizedLayer.ParametrizedLayer import ParametrizedLayer
from model_representation.EvoParam.IntEvoParam import IntEvoParam
from model_representation.EvoParam.FloatEvoParam import FloatEvoParam
from model_representation.EvoParam.CategEvoParam import CategEvoParam
from tensorflow.keras.initializers import Orthogonal
from tensorflow import keras
from keras.layers import ReLU

# TODO: changed value range
class LstmUnitsParam(IntEvoParam):
    _default_values = [8, 64, 256]
    _value_range = [2, 512]
    _key = "units"
    _mean = 256
    _sd = 300


class LstmDropoutParam(CategEvoParam):
    _default_values = [0.0, 0.1]
    _value_range = [0.0, 0.1, 0.15, 0.2, 0.25, 0.3]
    _weights = [0.5, 0.1, 0.1, 0.1, 0.1, 0.1]
    _key = "dropout"

class PLstmLayer(ParametrizedLayer):
    _layer = lambda **kwargs: keras.layers.LSTM(return_sequences=True, kernel_initializer=Orthogonal(), **kwargs)
    _param_classes = [LstmUnitsParam, LstmDropoutParam]
    _activation_function = ReLU