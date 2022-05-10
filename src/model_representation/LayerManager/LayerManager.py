import random
from model_representation.ParametrizedLayer.ParametrizedLayer import ParametrizedLayer 
import copy

class LayerManager():
    def __init__(self, default_layer_pool):
        self.default_layer_pool = default_layer_pool
    
    def get_random_default_layer(self) -> ParametrizedLayer: 
        return copy.deepcopy(random.choice(self.default_layer_pool))
