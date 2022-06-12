from model_representation.EvoParam.EvoParam import EvoParam
from model_representation.EvoParam.DEvoParam import DEvoParam
import random


class TupleCategDEvoParam(DEvoParam):

    def __init__(self, dependent_value, dependent_on_param) -> None:
        assert type(dependent_value) is tuple, "dependent_value must be tuple"
        assert dependent_value[0] in self._value_range, "dependent_value[0] must be of correct categorical"
        assert dependent_value[1] in self._value_range, "dependent_value[1] must be of correct categorical"
        assert not issubclass(type(dependent_on_param), DEvoParam), "dependent_on_param must be child of a normal EvoParam"

        super().__init__(dependent_value=dependent_value, dependent_on_param=dependent_on_param)
    
    def _mutated_tuple_value(self, mutation_percentage, tuple_pos):
        if random.random() < mutation_percentage:
            return self._distribution_val_sub_range()
        return self._dependent_value[tuple_pos]


    def mutate(self, intensity):
        """
        will only mutate the _dependent_value self.value() will get the real value on request
        """
        intensity_percentages = {
            "all": 1,
            "high": 0.2,
            "mid": 0.05,
            "low": 0.01
        }
        mutated_tuple_value_0 = self._mutated_tuple_value(mutation_percentage=intensity_percentages[intensity], tuple_pos=0)
        mutated_tuple_value_1 = self._mutated_tuple_value(mutation_percentage=intensity_percentages[intensity], tuple_pos=1)
        
        self._dependent_value = (mutated_tuple_value_0, mutated_tuple_value_1)
    
    # Lib -->
    def _distribution_val_sub_range(self):
        assert self._weights is not None, f"{self.__class__.__name__}: _weights must be set for _distribution_val_sub_range"
        return random.choices(self._value_range, weights=self._weights)[0]
    