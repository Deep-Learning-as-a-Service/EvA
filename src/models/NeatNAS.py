from neat import config, population, genome
import os

class NeatNAS:
    def __init__(self, n_generation, population_size, fitness):
        self.n_generation = n_generation
        self.population_size = population_size
        self.fitness = fitness

    def eval_fitness(self, population):
        for chromosome in population:
            self.fitness(chromosome)
    
    def run(self):
        config.Config().load(os.path.join('src','models','NeatNAS_config'))
        population.Population.evaluate = self.eval_fitness
        pop = population.Population()
        pop.epoch(100, report=True, save_best=True)
        winner = pop.stats[0][-1]
        print('Number of evaluations: %d' %winner.id)





