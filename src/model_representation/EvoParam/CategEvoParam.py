from model_representation.EvoParam.EvoParam import EvoParam
import random 

class CategEvoParam(EvoParam):

    def __init__(self, value):

        assert len(self._value_range) == len(self._weights), "_weights needs as many elements as _value_range"
        assert value in self._value_range, "value not possible"
        assert 0.99 < sum(self._weights) < 1.001 , "weights need to be in sum 1"

        # TODO: value_range checks
        return super().__init__(value=value)

    def mutate(self, intensity):

            intensity_percentages = {
                "all": 1,
                "high": 0.2,
                "mid": 0.05,
                "low": 0.01
            }
            mutation_percentage = intensity_percentages[intensity]
            if random.random() < mutation_percentage:
                self._value = self._distribution_val_sub_range()
        
    def _distribution_val_sub_range(self):
        assert self._weights is not None, f"{self.__class__.__name__}: _weights must be set for _distribution_val_sub_range"
        return random.choices(self._value_range, weights=self._weights)[0]