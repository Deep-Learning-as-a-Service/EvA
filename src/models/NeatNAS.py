from EvA.src.models.ModelGenome import ModelGenome
from neat import config, population, chromosome, genome, visualize
from neat.nn import nn_pure as nn


class NeatNAS:
    def __init__(self, n_generation, population_size):
        self._n_generation = n_generation
        self._population_size = population_size
        
    def run(fitness) -> ModelGenome:
        population.Population.evaluate = fitness
        pop = population.Population()
        pop.epoch(300, report=True, save_best=False)

        winner = pop.stats[0][-1]
        print('Number of evaluations: %d' %winner.id)