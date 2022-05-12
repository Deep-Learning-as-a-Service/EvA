from model_representation.ParametrizedLayer.ParametrizedLayer import ParametrizedLayer
from model_representation.EvoParam.IntEvoParam import IntEvoParam
from tensorflow import keras

class Conv2DFiltersParam(IntEvoParam):
    _default_values = [32, 64, 128]
    _value_range = [16, 128]
    _key = "filters"

# TODO: smart data specific choice?
# TODO: global dict? should not be bigger, than window size!
# - responsibility for correct params responsibility here or Model Checker
class Conv2DKernelSizeParam(TupleIntEvoParam):
    _default_values = [(1, 1), (3, 3), (5, 5)]
    _value_range = [(1, 1), (7, 7)]
    _key = "kernel_size"

# could be categorical! overlapping 50 percent 0 percent
# TODO: dependencies? reconfigure after mutation
class Conv2DStrideSizeParam(IntEvoParam):
    _default_values = [30, 70]
    _value_range = [1, 100]
    _key = "stride_size"


class PConv2DLayer(ParametrizedLayer):
    _layer = keras.layers.Conv1D
    _param_classes = [Conv2DFiltersParam, Conv2DKernelSizeParam, Conv2DStrideParam]

