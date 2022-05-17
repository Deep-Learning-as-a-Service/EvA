from model_representation.EvoParam.EvoParam import EvoParam
import random


class TupleIntEvoParam(EvoParam):
    
    def __init__(self, value):
        assert type(value) is tuple, "value must be of type tuple"
        assert type(value[0]) is int, "value[0] must be of type int"
        assert type(value[1]) is int, "value[1] must be of type int"
        assert type(self._value_range) is list, "value_range must be a list"
        assert type(self._value_range[0]) is tuple, "value_range[0] must be tuple"
        assert type(self._value_range[1]) is tuple, "value_range[1] must be tuple"
        # TODO: value_range checks
        return super().__init__(value=value)
    
    def _mutated_tuple_value(self, mutation_percentage, tuple_pos):
        value_tup_pos = self.value[tuple_pos]

        min_value_range = self._value_range[0][tuple_pos]
        max_value_range = self._value_range[1][tuple_pos]

        range_size = max_value_range - min_value_range
        min_limit = max(min_value_range, value_tup_pos - round(range_size * mutation_percentage))
        max_limit = min(max_value_range, value_tup_pos + round(range_size * mutation_percentage))
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
        mutated_tuple_value_0 = self._mutated_tuple_value(mutation_percentage=intensity_percentages[intensity], tuple_pos=0)
        mutated_tuple_value_1 = self._mutated_tuple_value(mutation_percentage=intensity_percentages[intensity], tuple_pos=1)
        
        self.value = (mutated_tuple_value_0, mutated_tuple_value_1)
    


"""
Implementation with EvoParam's

from abc import ABC, abstractmethod 
class TupleIntEvoParam(EvoParam):
    
    @classmethod
    def create(cls, value):
        assert type(value) is tuple, "value must be of type tuple"
        assert type(value[0]) intEvoParam, "value[0] must be of type IntEvoParam"
        assert type(value[1]) intEvoParam, "value[1] must be of type IntEvoParam"
        assert type(cls._value_range) is list, "value_range must be a list"
        assert type(cls._value_range[0]) is tuple, "value_range[0] must be tuple"
        assert type(cls._value_range[1]) is tuple, "value_range[1] must be tuple"
        # TODO: value_range checks
        return super().create(value=value)
    
    def _mutated_value(self, mutation_percentage):
        return (self.value[0].mutate(intensity=mutation_percentage), self.value[1].mutate(intensity=mutation_percentage))

    def mutate(self, intensity):
        intensity_percentages = {
            "all": 1,
            "high": 0.5,
            "mid": 0.15,
            "low": 0.05
        }
        
        self.value = self._mutated_value(mutation_percentage=intensity_percentages[intensity])
"""
    