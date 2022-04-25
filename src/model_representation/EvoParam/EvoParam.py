from abc import ABC, abstractmethod 

class EvoParam(ABC):
    def __init__(self, key, value, value_range) -> None:
        self.key = key
        self.value = value
        self.value_range = value_range # [min, max], ['categorical1', 'categorical2', ...]
    
    @abstractmethod
    def mutate(intensity):
        pass