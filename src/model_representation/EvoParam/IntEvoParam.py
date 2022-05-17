from model_representation.EvoParam.EvoParam import EvoParam
import random

from abc import ABC, abstractmethod 
class IntEvoParam(EvoParam):

    def __init__(self, value):
        assert type(value) is int, "value must be of type int"
        assert type(self._value_range) is list, "value_range must be a list"
        assert type(self._value_range[0]) is int, "value_range[0] must be int"
        assert type(self._value_range[1]) is int, "value_range[1] must be int"
        assert self._value_range[1] - self._value_range[0] > 0, "value_range needs to go from small to large"
        assert self._value_range[0] <= value and value <= self._value_range[1], "value out of range"

        super().__init__(value=value)
    
    def _mutated_value(self, mutation_percentage):
        range_size = self._value_range[1] - self._value_range[0]
        min_limit = max(self._value_range[0], self.value - round(range_size * mutation_percentage))
        max_limit = min(self._value_range[1], self.value + round(range_size * mutation_percentage))
        return random.randint(min_limit, max_limit)

    def mutate(self, intensity):
        """
        mutating IntEvoParams will have ranges for range params as followed:
        """
        intensity_percentages = {
            "all": 1,
            "high": 0.5,
            "mid": 0.15,
            "low": 0.05
        }
        
        self.value = self._mutated_value(mutation_percentage=intensity_percentages[intensity])
    

