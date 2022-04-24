import neat
import os
from nas.ModelGenome import ModelGenome

class NeatNAS:
    def __init__(self, n_generation, population_size, fitness):
        self.n_generation = n_generation
        self.population_size = population_size
        self.fitness = fitness
        
    def run(self) -> ModelGenome:
        # TODO: creation of ModelGenome with standard hyperparams - make this available in the experiment
        create_model_genome = lambda neat_genome: ModelGenome.create_with_default_params(neat_genome)

        def eval_genomes(neat_genomes, config):
            for neat_genome_id, neat_genome in neat_genomes:
                model_genome = create_model_genome(neat_genome)
                neat_genome.fitness = self.fitness(model_genome)
                print(neat_genome.fitness)

        # Load configuration.
        config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet, neat.DefaultStagnation, os.path.join(os.path.dirname(__file__), 'neat-nas-config'))

        # Create the population, which is the top-level object for a NEAT run.
        p = neat.Population(config)

        # Add a stdout reporter to show progress in the terminal.
        p.add_reporter(neat.StdOutReporter(False))

        # Run until a solution is found.
        winner_neat_genome = p.run(eval_genomes)

        return create_model_genome(winner_neat_genome)








