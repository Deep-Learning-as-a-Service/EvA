from model_representation.ParametrizedLayer.ParametrizedLayer import ParametrizedLayer
from model_representation.EvoParam.IntEvoParam import IntEvoParam
from model_representation.EvoParam.TupleIntEvoParam import TupleIntEvoParam
from model_representation.EvoParam.TupleCategDEvoParam import TupleCategDEvoParam
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

class Conv2DStridesParam(TupleCategDEvoParam):
    _dependent_class = Conv2DKernelSizeParam
    _default_values = [("step1", "step1"), ("100%", "step1")]
    _value_range = ["step1", "25%", "50%", "75%", "100%"]
    _key = "strides"

    def _value_tup_pos(self, tuple_position):
        dependent_value_tup_pos = self._dependent_value[tuple_position]
        if dependent_value_tup_pos == "step1":
            return 1
        else:
            percentage = int(dependent_value_tup_pos[:-1]) / 100 # "25%" -> 0.25
            # Conv2DKernelSizeParam is also a tuple
            return round(self._dependent_on_param.value[tuple_position] * percentage)

    @property
    def value(self):
        return (self._value_tup_pos(tuple_position=0), self._value_tup_pos(tuple_position=1))

class PConv2DLayer(ParametrizedLayer):
    _layer = keras.layers.Conv2D
    _param_classes = [Conv2DFiltersParam, Conv2DKernelSizeParam, Conv2DStridesParam]

