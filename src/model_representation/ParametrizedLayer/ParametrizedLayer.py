from abc import ABC, abstractmethod 
import keras.layers
import random

class ParametrizedLayer(ABC):
    innovation_number = 0
    intensity_to_param_mutation_probability = {
        "low": {
            "low" : 0.8,
            "mid" : 0.2,
            "high" : 0.0,
            "all" : 0.0
        },
        "mid": {
            "low" : 0.3,
            "mid" : 0.4,
            "high" : 0.3,
            "all" : 0.0
        },
        "high": {
            "low" : 0.0,
            "mid" : 0.3,
            "high" : 0.4,
            "all" : 0.3
        },
       "all": {
            "low" : 0.0,
            "mid" : 0.0,
            "high" : 0.0,
            "all" : 1.0
        }
    }
    def __init__(self, params) -> None:
        self.params = params
        self.innovation_number = ParametrizedLayer.innovation_number
        ParametrizedLayer.innovation_number += 1
    
    def get_func(self) -> keras.layers.Layer:
        kwargs = {}
        for param in self.params:
            kwargs[param._key] = param.value
        
        # lambda x: self.layer(**kwargs)(x)
        return self._layer(**kwargs)
    
    def mutate(self, layer_mutation_intensity):
        for param in self.params:
            param_mutation_intensity = ParametrizedLayer.get_param_mutation(layer_mutation_intensity)
            param.mutate(param_mutation_intensity)
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
    
    @classmethod
    def get_param_mutation(cls, intensity) -> str:
        rand_dict = cls.intensity_to_param_mutation_probability[intensity]
        assert sum(rand_dict.values()) == 1.0, "sum of probabilities should be 1"
        rand_number = random.random()
        mutation: str = None
        endProbability = 0.0
        for mutation_type, mutation_type_probability in rand_dict.items():
            endProbability += mutation_type_probability
            if rand_number <= endProbability:
                mutation = mutation_type
                break
        return mutation