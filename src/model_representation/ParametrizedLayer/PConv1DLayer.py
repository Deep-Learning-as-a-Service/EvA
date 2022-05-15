from model_representation.ParametrizedLayer.ParametrizedLayer import ParametrizedLayer
from model_representation.EvoParam.IntEvoParam import IntEvoParam
from tensorflow import keras

class Conv1DFiltersParam(IntEvoParam):
    _default_values = [32, 64, 128]
    _value_range = [16, 128]
    _key = "filters"

class Conv1DKernelSizeParam(IntEvoParam):
    _default_values = [2, 6, 12]
    _value_range = [2, 24]
    _key = "kernel_size"

#TODO make stride dependent from Kernel Size
class Conv1DStridesParam(IntEvoParam):
    _default_values = [1, 3, 5]
    _value_range = [1, 5]
    _key = "strides"
    


class PConv1DLayer(ParametrizedLayer):
    _layer = keras.layers.Conv1D
    _param_classes = [Conv1DFiltersParam, Conv1DKernelSizeParam, Conv1DStridesParam]

