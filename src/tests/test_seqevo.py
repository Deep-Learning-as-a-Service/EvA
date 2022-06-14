from optimizer.SeqEvo.SeqEvo import SeqEvo
from optimizer.SeqEvo.EvoTechniqueConfig import DefaultEvoTechniqueConfig
from optimizer.SeqEvo.Selector import Selector
from optimizer.SeqEvo.Crosser import Crosser
from utils.logger import logger
from utils.folder_operations import new_saved_experiment_folder
from model_representation.ParametrizedLayer.PDenseLayer import PDenseLayer
from model_representation.ParametrizedLayer.PConv1DLayer import PConv1DLayer
from model_representation.ParametrizedLayer.PConv2DLayer import PConv2DLayer
from model_representation.ParametrizedLayer.PLstmLayer import PLstmLayer
import utils.settings as settings
from optimizer.SeqEvo.SeqEvoHistory import SeqEvoHistory
import os
from optimizer.SeqEvo.InitialModelLayer import InitialModelLayer


layer_pool: 'list[ParametrizedLayer]' = [PConv2DLayer, PDenseLayer, PLstmLayer] #PConv1DLayer
settings.init(_layer_pool=layer_pool)


experiment_name = "TEST"
# Create Folder, save model export and evaluations there
experiment_folder_path = new_saved_experiment_folder(experiment_name) # create folder to store results
seqevo_history = SeqEvoHistory(
    path_to_file=os.path.join(experiment_folder_path, 'seqevo_history.csv')
)

parent_selector = Selector.select_from_fitness_probability
crossover_func = Crosser.middlepoint_crossover

def fitness_func(model_genome, log_func):
    return 0.5

model_genome = SeqEvo(
    n_generations = 300, 
    pop_size = 8,
    fitness_func = fitness_func,
    n_parents = 4,
    technique_config = DefaultEvoTechniqueConfig(),
    parent_selector=parent_selector,
    crossover_func=crossover_func,
    log_func=logger,
    seqevo_history=seqevo_history,
    initial_models = [
        InitialModelLayer.leander_deep_conv_1()
    ]
).run()