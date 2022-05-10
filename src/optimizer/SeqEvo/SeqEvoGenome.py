

import random
import utils.nas_settings as nas_settings
import model_representation.ParametrizedLayer as ParametrizedLayer
import model_representation.EvoParam as EvoParam
import utils.settings as settings

class SeqEvoGenome():
    _default_n_layers_range = [2, 7]
    """
    this is not the SeqEvoModelGenome! 
    This is a SeqEvo intern representation of a genome
    The other implementation is a generic ModelGenomeType with a network of ModelGenomes
    """

    def __init__(self, layers):
        self.layers = layers
        self.fitness = None

    @classmethod
    def create_random(cls):
        seq_evo_genome = cls.create_random_default()
        seq_evo_genome.mutate(intensity="all")
        return seq_evo_genome

    @classmethod
    def create_random_default(cls):
        size = random.randint(cls._default_n_layers_range[0], cls._default_n_layers_range[1])
        layers = []
        for _ in range(size):
            layer_class = random.choice(settings.layer_pool)
            layers.append(layer_class.create_random_default())
        return cls(layers=layers)



    
    def mutate(self, intensity):
        """
        TODO: 
        - add leave out layers
        - add change layer type
        """
        for layer in self.layers:
            layer.mutate(intensity)
        return self
