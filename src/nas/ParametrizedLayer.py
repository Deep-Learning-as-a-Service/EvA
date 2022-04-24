from abc import ABC, abstractmethod 
import keras.layers

class ParametrizedLayer(ABC):
    def __init__(self, layer, params) -> None:
        self.layer = layer
        self.params = params
    
    def get_func(self) -> keras.layers.Layer:
        kwargs = {}
        for param in self.params:
            kwargs[param.key] = param.value
        return self.layer(**kwargs)
    
    @abstractmethod
    def mutate(self, range):
        pass
    
    @abstractmethod
    def cross(self, parametrized_layer_02):
        pass