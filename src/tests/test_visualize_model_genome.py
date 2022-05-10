"""
Test visualize_genome
"""
import os
import neat

from model_representation.ModelGenome.ModelGenome import ModelGenome
import utils.nas_settings as nas_settings
import utils.settings as settings
from model_representation.ParametrizedLayer.PDenseLayer import PDenseLayer
from model_representation.EvoParam.IntEvoParam import IntEvoParam
from model_representation.LayerManager.LayerManager import LayerMapper
from tensorflow import keras
from model_representation.ModelGenome.ModelNode.ModelNode import ModelNode
from typing import Union
from visualization.visualize_neat_genome import visualize_neat_genome
from utils.folder_operations import new_saved_experiment_folder
import drawSvg as draw
from visualization.visualize_model_genome import visualize_model_genome

layer_pool = [

    PDenseLayer(keras.layers.Dense, 
                [IntEvoParam(
                    key="units", 
                    value=10, 
                    value_range=[5,50])
                ])
]
layer_mapper = LayerMapper(layer_pool=layer_pool)

settings.init()
nas_settings.init(layer_mapper)

experiment_folder_path = new_saved_experiment_folder('test visualize genome') # create folder to store results

def eval_genomes(neat_genomes, config):
    neat_genome = neat_genomes[0][1]
    model_genome = ModelGenome.create_with_default_params(neat_genome)

    visualize_neat_genome(config, neat_genome, os.path.join(experiment_folder_path, 'neat_genome'))
    visualize_model_genome(model_genome, os.path.join(experiment_folder_path, 'model_genome'))

    # visualize(model_genome, path)
    raise Exception("done")

# Load configuration.
config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet, neat.DefaultStagnation, os.path.join(os.path.dirname(__file__), 'neat-nas-config'))

# Create the population, which is the top-level object for a NEAT run.
p = neat.Population(config)

# Add a stdout reporter to show progress in the terminal.
p.add_reporter(neat.StdOutReporter(True))

p.run(eval_genomes)