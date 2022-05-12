from abc import ABC, abstractmethod 
import keras.layers

class ParametrizedLayer(ABC):
    innovation_number = 0
    def __init__(self, params) -> None:
        self.params = params
        self.innovation_number = ParametrizedLayer.innovation_number
        ParametrizedLayer.innovation_number += 1
    
    def __str__(self):
        return f"{self.__class__.__name__}({' '.join([str(param) for param in self.params])})"
    
    def get_func(self) -> keras.layers.Layer:
        kwargs = {}
        for param in self.params:
            kwargs[param._key] = param.value
        
        # lambda x: self.layer(**kwargs)(x)
        return self._layer(**kwargs)
    
    def mutate(self, intensity):
        for param in self.params:
            param.mutate(intensity)
        self.innovation_number = ParametrizedLayer.innovation_number
        ParametrizedLayer.innovation_number += 1
    
    @abstractmethod
    def cross(self, parametrized_layer_02):
        """
        subclass responsibility
        """
        raise NotImplementedError
    
    # Constructors ----------------------------------------------------------------------------------------------

    @classmethod
    def create_from_params(cls, params):
        assert cls._param_classes is not None, "_param_classes must be set"
        assert cls._layer is not None, "_layer must be set"

        for i, param in enumerate(params):
            param_class = cls._param_classes[i]
            assert type(param) is param_class, f"received param of type {type(param).__name__} must be of type {param_class.__name__}"
        return cls(params=params)
    
    @classmethod
    def create_random_default(cls):
        assert cls._param_classes is not None, "_param_classes must be set"
        assert cls._layer is not None, "_layer must be set"
        
        params = [param_class.create_random_default() for param_class in cls._param_classes]
        return cls(params=params)
    