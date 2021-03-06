
import numpy as np
import random
import utils.nas_settings as nas_settings
import model_representation.ParametrizedLayer as ParametrizedLayer
import model_representation.EvoParam as EvoParam
import utils.settings as settings
from utils.mutation_helper import get_key_from_prob_dict
import random
import copy

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
            "mutate_layer_type" : 0.2,
            "leave_out_layer" : 0.2,
            "mutate_layer_params" : 0.6
        },
        "high": {
            "mutate_layer_type" : 0.4,
            "leave_out_layer" : 0.0,
            "mutate_layer_params" : 0.6
        },
        "all": {
            "mutate_layer_type" : 0.0,
            "leave_out_layer" : 0.0,
            "mutate_layer_params" : 1
        }
    }
    """
    this is not the SeqEvoModelGenome! 
    This is a SeqEvo intern representation of a genome
    The other implementation is a generic ModelGenomeType with a network of ModelGenomes
    """

    def __init__(self, layers, created_from="-", parents=[]):
        self.layers = layers
        self.fitness = None
        self.created_from = created_from
        self.parents = None # parents
        self.parent_fitness = np.mean(list(map(lambda parent: parent.fitness, parents)))

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
        return cls(layers=layers, created_from="random_default")
    
    def layer_list_str(self):
        return ' '.join([str(layer) for layer in self.layers])
    
    def __str__(self):
        return f"SeqEvoGenome [{self.layer_list_str()}]\n\tfitness: {self.fitness}\n\tcreated_from: {self.created_from}"
    
    #TODO: get a better identifier by implementing the == operation on layer/param instead of using the logging function  
    def get_architecture_identifier(self):
        return self.layer_list_str()
    
    def mutate(self, intensity) -> 'SeqEvoGenome':

        mutated_seqevo_genome = copy.deepcopy(self)

        # Structural mutation
        struct_change_prob = mutated_seqevo_genome._mutation_intensitiy_to_struct_change[intensity]
        struct_change = get_key_from_prob_dict(struct_change_prob)
        if struct_change == "add_layer_random" and len(mutated_seqevo_genome.layers) < mutated_seqevo_genome._default_n_layers_range[1]:
            mutated_seqevo_genome.add_layer_random()
        elif struct_change == "remove_layer_random" and len(mutated_seqevo_genome.layers) > mutated_seqevo_genome._default_n_layers_range[0]:
            mutated_seqevo_genome.remove_layer_random()
        
        # Layer mutation
        for layer in mutated_seqevo_genome.layers:
            # get which mutation should be applied per layer
            layer_mutation_prob = SeqEvoGenome._mutation_intensity_to_layer_mutation[intensity]
            layer_mutation = get_key_from_prob_dict(layer_mutation_prob)

            if layer_mutation == "mutate_layer_type":
                layer = random.choice(settings.layer_pool).create_random_default()
            elif layer_mutation == "leave_out_layer":
                continue
            elif layer_mutation == "mutate_layer_params":
                layer.mutate(intensity)
        
        mutated_seqevo_genome.parent_fitness = mutated_seqevo_genome.fitness
        mutated_seqevo_genome.fitness = None
        mutated_seqevo_genome.created_from = f"mutate_{intensity}"
        return mutated_seqevo_genome

    def add_layer_random(self):
        layer_to_add = random.choice(settings.layer_pool).create_random_default()
        self.layers.insert(random.randint(0,len(self.layers)),layer_to_add)

    def remove_layer_random(self):
        layer_index_to_remove = random.randint(0, len(self.layers) - 1)
        self.layers.pop(layer_index_to_remove)