from model_representation.ParametrizedLayer.ParametrizedLayer import ParametrizedLayer
from model_representation.EvoParam.IntEvoParam import IntEvoParam
from tensorflow import keras

class Conv1DFiltersParam(IntEvoParam):
    _default_values = [32, 64, 128]
    _value_range = [16, 128]
    _key = "filters"

class Conv1DKernelSizeParam(IntEvoParam):
    _default_values = [2, 4]
    _value_range = [2, 4]
    _key = "kernel_size"
    


class PConv1DLayer(ParametrizedLayer):
    _layer = keras.layers.Conv1D
    _param_classes = [Conv1DFiltersParam, Conv1DKernelSizeParam]

