from abc import ABC, abstractmethod 
import random
from scipy.stats import truncnorm

class EvoParam(ABC):
    def __init__(self, value) -> None:
        assert value is not None, "value can't be None"
        assert self._default_values is not None, f"{self.__class__.__name__} has no cls._default_values"
        assert self._value_range is not None, f"{self.__class__.__name__} has no cls._value_range"
        assert self._key is not None, f"{self.__class__.__name__} has no cls._key"
        assert self._distribution_val_sub_range is not None, f"{self.__class__.__name__} has no cls._distribution_val_sub_range"

        self._value = value
    
    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, value):
        assert value is not None, "value can't be None"
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

    # Lib -->
    @staticmethod
    def normal_distribution_val_sub_range_func(mean, sd, apply_to_float):
        def get_truncated_normal(mean, sd, low, upp):
            return truncnorm(
                (low - mean) / sd, (upp - mean) / sd, loc=mean, scale=sd)

        def distribution_val_sub_range(low, upp, number_of_samples=1):
            if low == upp: 
                if number_of_samples == 1:
                    return low
                else:
                    return [low for _ in range(number_of_samples)]
                    
            output = get_truncated_normal(mean=mean, sd=sd, low=low, upp=upp).rvs(number_of_samples)
            output = list(map(apply_to_float, output))
            if number_of_samples == 1:
                return output[0]
            return output
        return distribution_val_sub_range
