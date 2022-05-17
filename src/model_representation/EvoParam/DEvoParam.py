from model_representation.EvoParam.EvoParam import EvoParam
import random

class DEvoParam(EvoParam):
    """
    Dependent Evolutionary Parameter
    - has no normal self.value (will calculate it from dependent_value, dependent_on_param, if you call value())
    - will mutate the dependent_value only
    """
    def __init__(self, dependent_value, dependent_on_param) -> None:
        assert issubclass(self._dependent_class, EvoParam), f"{self.__class__.__name__} has no cls._dependent_class of type EvoParam"
        assert type(dependent_on_param) is self._dependent_class, f"dependent_on_param must be of type {self._dependent_class.__name__}"
        
        self._dependent_value = dependent_value
        self._dependent_on_param = dependent_on_param
        super().__init__(value=self.value)
    
    @property
    def value(self):
        """
        subclass responsibility!
        calculate from dependent_value and dependent_on_param
        """
        raise NotImplementedError

    @value.setter
    def value(self, value):
        raise Exception("cannot set value directly for Dependent Evolutionary Parameter")
    
    @classmethod
    def create_random_default(cls, dependent_on_param):
        return cls.create(dependent_value=random.choice(cls._default_values), dependent_on_param=dependent_on_param)
    
    @classmethod
    def create(cls, dependent_value, dependent_on_param):
        return cls(dependent_value=dependent_value, dependent_on_param=dependent_on_param)
