from model_representation.EvoParam.CategEvoParam import CategEvoParam
from model_representation.ParametrizedLayer.ParametrizedLayer import ParametrizedLayer
from model_representation.EvoParam.IntEvoParam import IntEvoParam
from model_representation.EvoParam.FloatEvoParam import FloatEvoParam
from model_representation.EvoParam.TupleIntEvoParam import TupleIntEvoParam
from model_representation.EvoParam.TupleCategDEvoParam import TupleCategDEvoParam
from model_representation.EvoParam.BoolEvoParam import BoolEvoParam

from tensorflow.keras.initializers import Orthogonal
from tensorflow import keras
from keras.layers import Dropout, MaxPool2D
from keras.layers import SpatialDropout2D, BatchNormalization
from keras.layers import ReLU


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


class Conv2DMaxPoolParam(CategEvoParam):
    """
    after layer param!
    - will not get passed in the _layer function from the PLayer
    - will create its own layer, therefore has a _layer() method
    """
    _default_values = ["None", (2,2)]
    _value_range = ["None", (2,2), (4,4)]
    _weights = [0.5, 0.3, 0.2]
    _key = "max_pooling"

    def _layer(self):
        """
        function that creates a tensor to tensor func from a Conv2DMaxPoolParam
        """
        def max_pooling(tensor):
            if self._value != "None":
                return MaxPool2D(pool_size=self._value)(tensor)
            return tensor
        return max_pooling

class Conv2DDropoutParam(CategEvoParam):
    _default_values = [0.0, 0.1]
    _value_range = [0.0, 0.1, 0.15, 0.2, 0.25, 0.3]
    _weights = [0.5, 0.1, 0.1, 0.1, 0.1, 0.1]
    _key = "dropout"

    # TODO: could be lambda as well... change in get_func()!
    # _layer = lambda tensor: Dropout(rate=self._value)(tensor) if self._value != 0 else tensor # function that returns 

    def _layer(self):
        def dropout(tensor):
            if self._value != 0:
                return SpatialDropout2D(rate=self._value, data_format="channels_last")(tensor)
            return tensor
        return dropout

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

class Conv2DBatchNormalizationParam(BoolEvoParam):
    _default_values = [True, False]
    _weights = [0.2, 0.8]
    _key = "batch_normalization"

    def _layer(self):
        def batch_normalization(tensor):
            if self.value:
                return BatchNormalization()(tensor)
            return tensor
        return batch_normalization



class PConv2DLayer(ParametrizedLayer):
    _layer = lambda **kwargs: keras.layers.Conv2D(kernel_initializer=Orthogonal(), **kwargs)
    _param_classes = [Conv2DFiltersParam, Conv2DKernelSizeParam, Conv2DStridesParam, Conv2DDropoutParam, Conv2DMaxPoolParam, Conv2DBatchNormalizationParam]
    _after_layer_params = [Conv2DBatchNormalizationParam, Conv2DMaxPoolParam, Conv2DDropoutParam]
    _activation_function = ReLU
    

