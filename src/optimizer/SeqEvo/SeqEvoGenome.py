

import random
import utils.nas_settings as nas_settings
import model_representation.ParametrizedLayer as ParametrizedLayer
import model_representation.EvoParam as EvoParam
import utils.settings as settings
from utils.mutation_helper import get_key_from_prob_dict
import random

class SeqEvoGenome():
    _default_n_layers_range = [2, 7]
    _mutation_intensitiy_to_struct_change = {
        "low" : {
            "add_layer_random" : 0,
            "remove_layer_random": 0,
            "none": 1
        },
        "mid" : {
            "add_layer_random" : 0.1,
            "remove_layer_random": 0.1,
            "none": 0.8
        },
        "high" : {
            "add_layer_random" : 0.25,
            "remove_layer_random": 0.25,
            "none": 0.5
        },
        "all" : {
            "add_layer_random" : 0.5,
            "remove_layer_random": 0.5,
            "none": 0.0
        } 
    }

    # low: at one point high param mutation? or at many places low
    _mutation_intensity_to_layer_mutation = {
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
            "mutate_layer_type" : 0.3,
            "leave_out_layer" : 0.2,
            "mutate_layer_params" : 0.5
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

        # Structural mutation
        struct_change_prob = self._mutation_intensitiy_to_struct_change[intensity]
        struct_change = get_key_from_prob_dict(struct_change_prob)
        if struct_change == "add_layer_random":
            self.add_layer_random()
        elif struct_change == "remove_layer_random":
            self.remove_layer_random()
        
        # Layer mutation
        for layer in self.layers:
            # get which mutation should be applied per layer
            layer_mutation_prob = SeqEvoGenome._mutation_intensity_to_layer_mutation[intensity]
            layer_mutation = get_key_from_prob_dict(layer_mutation_prob)

            if layer_mutation == "mutate_layer_type":
                layer = random.choice(settings.layer_pool).create_random_default()
            elif layer_mutation == "leave_out_layer":
                continue
            elif layer_mutation == "mutate_layer_params":
                layer.mutate(intensity)
        return self

    def add_layer_random(self):
        layer_to_add = random.choice(settings.layer_pool).create_random_default()
        self.layers.insert(random.randint(0,len(self.layers)),layer_to_add)

    def remove_layer_random(self):
        layer_index_to_remove = random.randint(0, len(self.layers) - 1)
        self.layers.pop(layer_index_to_remove)