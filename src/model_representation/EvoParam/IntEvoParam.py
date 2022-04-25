from model_representation.EvoParam.EvoParam import EvoParam
import random

class IntEvoParam(EvoParam):
    
    def __init__(self, key, value, value_range):
        super().__init__(key, value, value_range)
        assert type(self.value) is int
        assert type(self.value_range) is list
        assert type(self.value_range[0]) is int
        assert type(self.value_range[1]) is int
        assert self.value_range[1] - self.value_range[0] > 0
        
    def mutate(self, intensity):
        """
        mutating IntEvoParams will have ranges for range params as followed:
        "all" => whole range
        "high" => +/- 50% of range
        "mid" => +/- 15% of range
        "low" => +/- 5% of range
        """
        # TODO: Refactor calculation - one function that takes percentage and returns value
        range_size = self.value_range[1] - self.value_range[0]
        if (intensity == "all"):
            self.value = random.randint(self.value_range[0],self.value_range[1])
            
        elif (intensity == "high"):
            min_limit = max(self.value_range[0], self.value - (range_size * 0.5))
            max_limit = min(self.value_range[1], self.value + (range_size * 0.5))
            self.value = random.randint(min_limit, max_limit)
            
        elif (intensity == "mid"):
            min_limit = max(self.value_range[0], self.value - (range_size * 0.15))
            max_limit = min(self.value_range[1], self.value + (range_size * 0.15))
            self.value = random.randint(min_limit, max_limit)
            
        elif (intensity == "low"):
            min_limit = max(self.value_range[0], self.value - (range_size * 0.05))
            max_limit = min(self.value_range[1], self.value + (range_size * 0.05))
            self.value = random.randint(min_limit, max_limit)
            
        else:
            raise Exception("Unknown intensity Param")