

import random
import utils.nas_settings as nas_settings
import model_representation.ParametrizedLayer as ParametrizedLayer
import model_representation.EvoParam as EvoParam

class SeqEvoGenome():

    @staticmethod
    def create_random(size = 5):
        layers = []
        for _ in range(size):
            
            # choose a random layer from pool and randomize all layer params
            layer = nas_settings.layer_mapper.generate_random_layer()
            layer.mutate("all")
            layers.append(layer)
            
        return SeqEvoGenome(layers)
        
    def __init__(self, layers):
        self.layers = layers
        self.fitness = None
    
    def mutate(self, intensity):
        for layer in self.layers:
            layer.mutate(intensity)
        return self
