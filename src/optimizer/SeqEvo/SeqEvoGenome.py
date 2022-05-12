

import random
import utils.nas_settings as nas_settings
import model_representation.ParametrizedLayer as ParametrizedLayer
import model_representation.EvoParam as EvoParam
import utils.settings as settings
from utils.mutation_helper import get_mutation_probability
import random

class SeqEvoGenome():
    _default_n_layers_range = [2, 7]
    _intensity_to_layer_probabilities = {
        "low": {
            "mutate_layer_type" : 0.0,
            "leave_out_layer" : 0.5,
            "mutate_layer_params" : 0.5
        },
        "mid": {
            "mutate_layer_type" : 0.05,
            "leave_out_layer" : 0.35,
            "mutate_layer_params" : 0.6
        },
        "high": {
            "mutate_layer_type" : 0.1,
            "leave_out_layer" : 0.2,
            "mutate_layer_params" : 0.7
        },
        "all": {
            "mutate_layer_type" : 0.3,
            "leave_out_layer" : 0.0,
            "mutate_layer_params" : 0.7
        }
    }
    """
    this is not the SeqEvoModelGenome! 
    This is a SeqEvo intern representation of a genome
    The other implementation is a generic ModelGenomeType with a network of ModelGenomes
    """

    def __init__(self, layers, created_from="-"):
        self.layers = layers
        self.fitness = None
        self.created_from = created_from

    @classmethod
    def create_random(cls):
        seq_evo_genome = cls.create_random_default()
        seq_evo_genome.mutate(intensity="all")
        seq_evo_genome.created_from = "random"
        return seq_evo_genome

    @classmethod
    def create_random_default(cls):
        size = random.randint(cls._default_n_layers_range[0], cls._default_n_layers_range[1])
        layers = []
        for _ in range(size):
            layer_class = random.choice(settings.layer_pool)
            layers.append(layer_class.create_random_default())
        return cls(layers=layers, created_from="random_default")
    
    def __str__(self):
        return f"SeqEvoGenome [{' '.join([str(layer) for layer in self.layers])}]\n\tfitness: {self.fitness}\n\tcreated_from: {self.created_from}"
    
    def mutate(self, intensity):
        """
        TODO: 
        - add leave out layers
        - add change layer type
        """
        for layer in self.layers:
            # get which mutation should be applied per layer
            prob_dict = SeqEvoGenome._intensity_to_layer_probabilities[intensity]
            mutation = get_mutation_probability(prob_dict)

            if mutation == "mutate_layer_type":
                #TODO: mutate layer type
                layer = random.choice(settings.layer_pool).create_random_default()
            elif mutation == "leave_out_layer":
                continue
            elif mutation == "mutate_layer_params":
                layer.mutate(intensity)
        return self

    def add_node_random(self):
        layer_to_add = random.choice(settings.layer_pool).create_random_default()
        self.layers.insert(random.randint(0,len(self.layers)),layer_to_add)

    def remove_node_random(self):
        layer_index_to_remove = random.randint(0,len(self.layers))
        self.layers.pop(layer_index_to_remove)