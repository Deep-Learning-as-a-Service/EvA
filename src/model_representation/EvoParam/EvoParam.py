from abc import ABC, abstractmethod 
import random

class EvoParam(ABC):
    def __init__(self, value) -> None:
        assert self._default_values is not None, f"{self.__class__.__name__} has no cls._default_values"
        assert self._value_range is not None, f"{self.__class__.__name__} has no cls._value_range"
        assert self._key is not None, f"{self.__class__.__name__} has no cls._key"

        self._value = value
    
    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, value):
        self._value = value
    
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
        return cls.create(value=random.choice(cls._default_values))
    
    @classmethod
    def create(cls, value):
        return cls(value=value)
