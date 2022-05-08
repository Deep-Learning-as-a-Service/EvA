import random
from model_representation.ParametrizedLayer.ParametrizedLayer import ParametrizedLayer 
import copy

class LayerMapper():
    def __init__(self, layer_pool):
        self.layer_pool = layer_pool
        self.neat_node_key_to_layer = {}

    def get_layer(self, neat_node_key) -> ParametrizedLayer:
        if neat_node_key in list(self.neat_node_key_to_layer.keys()):
            return self.neat_node_key_to_layer[neat_node_key]
        
        layer = self.generate_random_layer()
        self.neat_node_key_to_layer[neat_node_key] = layer
        return layer
    
    def generate_random_layer(self) -> ParametrizedLayer: 
        layer = copy.deepcopy(random.choice(self.layer_pool))
        
        # mutate "all" to change innovation number of layer and to randomize layer params in param space
        layer.mutate("all")
        return layer
