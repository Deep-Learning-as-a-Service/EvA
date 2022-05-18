from model_representation.ParametrizedLayer.ParametrizedLayer import ParametrizedLayer
from model_representation.EvoParam.IntEvoParam import IntEvoParam
from model_representation.EvoParam.CategDEvoParam import CategDEvoParam
from tensorflow import keras

class Conv1DFiltersParam(IntEvoParam):
    _default_values = [32, 64, 128]
    _value_range = [16, 128]
    _key = "filters"

class Conv1DKernelSizeParam(IntEvoParam):
    _default_values = [2, 6, 12]
    _value_range = [2, 24]
    _key = "kernel_size"

class Conv1DStridesParam(CategDEvoParam):
    """
    TODO optional:
    - mutate should go 1 label to the right/ to the left, instead of coinflip on everything
    """

    _dependent_class = Conv1DKernelSizeParam
    _default_values = ["step1", "50%", "100%"]
    _value_range = ["step1", "25%", "50%", "75%", "100%"]
    _key = "strides"

    @property
    def value(self):
        if self._dependent_value == "step1":
            return 1
        else:
            percentage = int(self._dependent_value[:-1]) / 100 # "25%" -> 0.25
            # Conv2DKernelSizeParam is also a tuple
            return max(1, round(self._dependent_on_param.value * percentage))


class PConv1DLayer(ParametrizedLayer):
    _layer = keras.layers.Conv1D
    _param_classes = [Conv1DFiltersParam, Conv1DKernelSizeParam, Conv1DStridesParam]

