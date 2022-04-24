from abc import ABC, abstractmethod 

class EvoParam(ABC):
    def __init__(self, key, value, range):
        self.key = key
        self.value = value
        self.range = range
    
    @abstractmethod
    def mutate(range):
        pass