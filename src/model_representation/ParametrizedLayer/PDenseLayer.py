from model_representation.ParametrizedLayer.ParametrizedLayer import ParametrizedLayer
import random
from tensorflow import keras
from model_representation.EvoParam.IntEvoParam import IntEvoParam

class DenseUnitsParam(IntEvoParam):
    _default_values = [32, 64, 128]
    _value_range = [16, 128]
    _key = "units"
    _mean = 64
    _sd = 20

class PDenseLayer(ParametrizedLayer):
    _layer = keras.layers.Dense
    _param_classes = [DenseUnitsParam]
            