from model_representation.EvoParam.EvoParam import EvoParam
from model_representation.EvoParam.DEvoParam import DEvoParam
import random


class BoolEvoParam(EvoParam):
    _value_range = [True, False]

    def __init__(self, value) -> None:
        assert value in self._value_range, "value not possible"
        assert sum(self._weights) == 1, "_weights in sum need to be 1"
        assert len(self._weights) == len(self._value_range), "number of weights need to match value_range"

        super().__init__(value=value)

    def mutate(self, intensity):
        """
        will only mutate the _dependent_value self.value() will get the real value on request
        """
        intensity_percentages = {
            "all": 1,
            "high": 0.4,
            "mid": 0.2,
            "low": 0.05
        }
        mutation_percentage = intensity_percentages[intensity]
        if random.random() < mutation_percentage:
            self._value = random.choices(self._value_range, weights=self._weights)[0]
    