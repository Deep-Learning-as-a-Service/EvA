class NeatNASLayerManager():

    def __init__(self, layer_pool):
        super().__init__(layer_pool=layer_pool)
        self.neat_node_key_to_layer = {}

    def get_layer(self, neat_node_key) -> ParametrizedLayer:
        if neat_node_key in list(self.neat_node_key_to_layer.keys()):
            return self.neat_node_key_to_layer[neat_node_key]
        
        layer = self.generate_random_layer()
        self.neat_node_key_to_layer[neat_node_key] = layer
        return layer