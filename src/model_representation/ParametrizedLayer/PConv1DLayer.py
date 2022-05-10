from model_representation.ParametrizedLayer.ParametrizedLayer import ParametrizedLayer
from model_representation.EvoParam.IntEvoParam import IntEvoParam
import random
from tensorflow import keras

class ConvFiltersParam(IntEvoParam):
    _default_values = [32, 64, 128]
    _value_range = [16, 128]
    _key = "filters"

class ConvKernelSizeParam(IntEvoParam):
    _default_values = [2, 4]
    _value_range = [2, 4]
    _key = "kernel_size"
    


class PConv1DLayer(ParametrizedLayer):
    _layer = keras.layers.Conv1D
    _param_classes = [ConvFiltersParam, ConvKernelSizeParam]

    def cross(self, parametrized_layer_02):
        crossed_params = []
        for idx, param in enumerate(self.params):
            
            # make sure that all params of the layers match
            assert param.key == parametrized_layer_02.params[idx].key
            
            # get correct subclass of EvoParam and build new instance
            ParamClass = param.__class__
            
            # simple coinflip whose value to choose from (TODO: maybe weight via parents accuracy)
            value = param.value if round(random.random()) == 1 else parametrized_layer_02.params[idx].value
            
            # add crossed param to list 
            crossed_params += ParamClass(key=param.key, value=value, range=param.range) 
            
        return PConvLayer(layer=self.layer, params=crossed_params)
