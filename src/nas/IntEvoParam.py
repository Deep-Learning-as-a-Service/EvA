from nas.EvoParam import EvoParam
import random

class IntEvoParam(EvoParam):
    
    def __init__(self, key, value, range):
        super().__init__(key, value, range)
        assert type(self.value) is int
        assert type(self.range) is list
        assert type(self.range[0]) is int
        assert type(self.range[1]) is int
        assert self.range[1] - self.range[0] > 0
        
    def mutate(self, range):
        """
        mutating IntEvoParams will have ranges for range params as followed:
        "all" => whole range
        "high" => +/- 50% of range
        "mid" => +/- 15% of range
        "low" => +/- 5% of range
        """
        range_size = self.range[1] - self.range[0]
        if(range == "all"):
            self.value = random.randint(self.range[0],self.range[1])
            
        elif(range == "high"):
            min_limit = max(self.range[0], self.value - (range_size * 0.5))
            max_limit = min(self.range[1], self.value + (range_size * 0.5))
            self.value = random.randint(min_limit, max_limit)
            
        elif(range == "mid"):
            min_limit = max(self.range[0], self.value - (range_size * 0.15))
            max_limit = min(self.range[1], self.value + (range_size * 0.15))
            self.value = random.randint(min_limit, max_limit)
            
        elif(range == "low"):
            min_limit = max(self.range[0], self.value - (range_size * 0.05))
            max_limit = min(self.range[1], self.value + (range_size * 0.05))
            self.value = random.randint(min_limit, max_limit)
            
        else:
            raise Exception("Unknown Range Param")