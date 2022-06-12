from model_representation.EvoParam.EvoParam import EvoParam
from model_representation.EvoParam.DEvoParam import DEvoParam
import random


class CategDEvoParam(DEvoParam):

    def __init__(self, dependent_value, dependent_on_param) -> None:
        assert type(dependent_value) is str, "dependent_value must be tuple"
        assert dependent_value in self._value_range, "dependent_value[0] must be of correct categorical"
        assert not issubclass(type(dependent_on_param), DEvoParam), "dependent_on_param must be child of a normal EvoParam"

        super().__init__(dependent_value=dependent_value, dependent_on_param=dependent_on_param)

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
        mutation_percentage = intensity_percentages[intensity]
        if random.random() < mutation_percentage:
            self._dependent_value = self._distribution_val_sub_range()
    
    def _distribution_val_sub_range(self):
        assert self._weights is not None, f"{self.__class__.__name__}: _weights must be set for _distribution_val_sub_range"
        return random.choices(self._value_range, weights=self._weights)[0]
    