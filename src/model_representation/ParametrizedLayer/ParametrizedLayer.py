from abc import ABC, abstractmethod 
import keras.layers

class ParametrizedLayer(ABC):
    innovation_number = 0
    def __init__(self, layer, params) -> None:
        self.layer = layer
        self.params = params
        self.innovation_number = ParametrizedLayer.innovation_number
        ParametrizedLayer.innovation_number += 1
    
    def get_func(self) -> keras.layers.Layer:
        kwargs = {}
        for param in self.params:
            kwargs[param.key] = param.value
        
        # lambda x: self.layer(**kwargs)(x)
        return self.layer(**kwargs)
    
    def mutate(self, intensity):
        for param in self.params:
            param.mutate(intensity)
        self.innovation_number = ParametrizedLayer.innovation_number
        ParametrizedLayer.innovation_number += 1
    
    @abstractmethod
    def cross(self, parametrized_layer_02):
        pass