from abc import ABC, abstractmethod 
import keras.layers
from utils.mutation_helper import get_key_from_prob_dict
from model_representation.EvoParam.DEvoParam import DEvoParam

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
        assert self._layer is not None, f"{self.__class__.__name__} has no cls._layer"
        assert self._param_classes is not None, f"{self.__class__.__name__} has no cls._param_classes"

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
    
    def mutate(self, intensity: str) -> None:
        for param in self.params:
            param_mutation_prob = self.intensity_to_param_mutation_probability[intensity]
            param_mutation = get_key_from_prob_dict(param_mutation_prob)
            param.mutate(param_mutation)
        self.innovation_number = ParametrizedLayer.innovation_number
        ParametrizedLayer.innovation_number += 1
    
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
        
        # create all NOT DEPENDENT params
        params = []
        for param_class in cls._param_classes:
            if not issubclass(param_class, DEvoParam):
                params.append(param_class.create_random_default())

        # create all DEPENDENT params
        for param_class in cls._param_classes:
            if issubclass(param_class, DEvoParam):
                # find dependent param, needed for creation (hopefully already created)
                params_of_dependent_class = list(filter(lambda param: type(param) is param_class._dependent_class, params))
                assert len(params_of_dependent_class) == 1, f"there must be exactly one param of type {param_class._dependent_class.__name__} to create a {param_class.__name__}"
                dependent_on_param = params_of_dependent_class[0]

                params.append(param_class.create_random_default(dependent_on_param=dependent_on_param))

        return cls(params=params)
    