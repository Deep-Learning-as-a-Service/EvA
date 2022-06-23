from model_representation.ParametrizedLayer.ParametrizedLayer import ParametrizedLayer
import random
from tensorflow import keras
from model_representation.EvoParam.IntEvoParam import IntEvoParam

# TODO: changed value range
class DenseUnitsParam(IntEvoParam):
    _default_values = [32, 64, 128, 256, 512]
    _value_range = [16, 1024]
    _key = "units"
    _mean = 514
    _sd = 2000

class PDenseLayer(ParametrizedLayer):
    _layer = keras.layers.Dense
    _param_classes = [DenseUnitsParam]
            