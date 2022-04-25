import os
import neat
from model_representation.ModelGenome import ModelGenome
import utils.nas_settings as nas_settings
from model_representation.ParametrizedLayer.PDenseLayer import PDenseLayer
from model_representation.EvoParam.IntEvoParam import IntEvoParam
from model_representation.LayerMapper import LayerMapper
from tensorflow import keras

layer_pool = [

    PDenseLayer(keras.layers.Dense, 
                [IntEvoParam(
                    key="units", 
                    value=10, 
                    value_range=[5,50])
                ])
]

layer_mapper = LayerMapper(layer_pool=layer_pool)
nas_settings.init(layer_mapper)

def eval_genomes(neat_genomes, config):
    model_genome = ModelGenome.create_with_default_params(neat_genomes[0][1])
    print(model_genome)
    # visualize(model_genome, path)
    raise Exception("done")

# Load configuration.
config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet, neat.DefaultStagnation, os.path.join(os.path.dirname(__file__), 'neat-nas-config'))

# Create the population, which is the top-level object for a NEAT run.
p = neat.Population(config)

# Add a stdout reporter to show progress in the terminal.
p.add_reporter(neat.StdOutReporter(True))

# Run until a solution is found.
try:
    p.run(eval_genomes)
except Exception as e:
    pass