from abc import ABC, abstractmethod 
import random

class EvoParam(ABC):
    def __init__(self, value) -> None:
        self.value = value
    
    def __str__(self):
        return f"{self._key}={self.value}"
    
    @abstractmethod
    def mutate(self, intensity):
        """
        subclass responsibility
        """
        raise NotImplementedError
    
    @classmethod
    def create_random_default(cls):
        assert cls._default_values is not None, "random_default_values must be set"
        return cls.create(value=random.choice(cls._default_values))
    
    @classmethod
    def create(cls, value):
        assert cls._default_values is not None, "default_values must be set"
        assert cls._value_range is not None, "value_range must be set"
        assert cls._key is not None, "key must be set"
        return cls(value=value)
