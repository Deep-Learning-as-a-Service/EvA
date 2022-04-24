import random
from nas.ParametrizedLayer import ParametrizedLayer 
from nas.PDenseLayer import PDenseLayer

class LayerMapper():
    def __init__(self, layer_pool):
        self.layer_pool = layer_pool
        self.neat_node_key_to_layer = {}

    def get_layer(self, neat_node_key) -> ParametrizedLayer:
        if neat_node_key in list(self.neat_node_key_to_layer.keys()):
            return self.neat_node_key_to_layer[neat_node_key]
        
        layer = self._get_random_layer()
        self.neat_node_key_to_layer[neat_node_key] = layer
        return layer
    
    def _get_random_layer(self) -> ParametrizedLayer:
        return random.choice(self.layer_pool)
