from model_representation.ParametrizedLayer.ParametrizedLayer import ParametrizedLayer
from model_representation.EvoParam.IntEvoParam import IntEvoParam
from model_representation.EvoParam.TupleIntEvoParam import TupleIntEvoParam
from model_representation.EvoParam.TupleCategDEvoParam import TupleCategDEvoParam
from tensorflow.keras.initializers import Orthogonal
from tensorflow import keras

# TODO: changed value range
class Conv2DFiltersParam(IntEvoParam):
    _default_values = [32, 64, 128]
    _value_range = [16, 256]
    _key = "filters"
    _mean = 64
    _sd = 200

# TODO: smart data specific choice?
# TODO: global dict? should not be bigger, than window size!
# - responsibility for correct params responsibility here or Model Checker
class Conv2DKernelSizeParam(TupleIntEvoParam):
    _default_values = [(1, 1), (3, 3), (5, 5)]
    _value_range = [(1, 1), (10, 10)]
    _key = "kernel_size"
    _mean = 3
    _sd = 8
    

class Conv2DStridesParam(TupleCategDEvoParam):
    """
    TODO
    - mutate should go 1 label to the right/ to the left, instead of coinflip on everything
    """
    _dependent_class = Conv2DKernelSizeParam
    _default_values = [("step1", "step1"), ("100%", "step1")]
    _value_range = ["step1", "25%", "50%", "75%", "100%"]
    _weights = [0.6, 0.1, 0.1, 0.1, 0.1]
    _key = "strides"

    def _value_tup_pos(self, tuple_position):
        dependent_value_tup_pos = self._dependent_value[tuple_position]
        if dependent_value_tup_pos == "step1":
            return 1
        else:
            percentage = int(dependent_value_tup_pos[:-1]) / 100 # "25%" -> 0.25
            # Conv2DKernelSizeParam is also a tuple
            return max(1, round(self._dependent_on_param.value[tuple_position] * percentage))

    @property
    def value(self):
        return (self._value_tup_pos(tuple_position=0), self._value_tup_pos(tuple_position=1))

class PConv2DLayer(ParametrizedLayer):
    _layer = lambda **kwargs: keras.layers.Conv2D(activation="relu", kernel_initializer=Orthogonal(), **kwargs)
    _param_classes = [Conv2DFiltersParam, Conv2DKernelSizeParam, Conv2DStridesParam]

