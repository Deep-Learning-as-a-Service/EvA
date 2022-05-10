from model_representation.ParametrizedLayer.ParametrizedLayer import ParametrizedLayer
import random
from tensorflow import keras
from model_representation.EvoParam.IntEvoParam import IntEvoParam

class DenseUnitsParam(IntEvoParam):
    _default_values = [32, 64, 128]
    _value_range = [16, 128]
    _key = "units"

class PDenseLayer(ParametrizedLayer):
    _layer = keras.layers.Dense
    _param_classes = [DenseUnitsParam]

    def cross(self, parametrized_layer_02: 'PDenseLayer'):
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
            
        return PDenseLayer(layer=self.layer, params=crossed_params)
            
            