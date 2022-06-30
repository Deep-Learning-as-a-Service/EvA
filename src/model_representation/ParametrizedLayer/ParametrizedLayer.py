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
        main_layer_params = self.params if not hasattr(self, '_after_layer_params') else [p for p in self.params if p.__class__ not in self._after_layer_params]
        for param in main_layer_params:
            kwargs[param._key] = param.value
        
        main_layer_func = self.__class__._layer(**kwargs) # this is a tensor to tensor function

        if hasattr(self, '_after_layer_params'):
            def param_tensor_func(after_layer_param_class):
                after_layer_params_matching = [param for param in self.params if param.__class__ == after_layer_param_class]
                assert len(after_layer_params_matching) == 1, f"after_layer_param_class {after_layer_param_class} has not exactly one match in {self.__class__}"
                after_layer_param = after_layer_params_matching[0]
                return after_layer_param._layer()
            
            # weird coding because lambda recursive naming problem | fun = lambda a: a+1 | fun = lambda a: fun(a) + 4 | appending functions not possible
            if len(self._after_layer_params) == 1:
                return lambda tensor: param_tensor_func(self._after_layer_params[0])(main_layer_func(tensor))
            elif len(self._after_layer_params) == 2:
                return lambda tensor: param_tensor_func(self._after_layer_params[1])(param_tensor_func(self._after_layer_params[0])(main_layer_func(tensor)))
            elif len(self._after_layer_params) == 3:
                return lambda tensor: param_tensor_func(self._after_layer_params[2])(param_tensor_func(self._after_layer_params[1])(param_tensor_func(self._after_layer_params[0])(main_layer_func(tensor))))

            else:
                raise Exception(f"found more _after_layer_params than implemented")
        
        return main_layer_func
    
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
    