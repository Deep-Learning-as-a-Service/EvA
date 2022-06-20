"""
HyPaEvo(
    input_model_genome=model_genome,
    n_generations = 300,
    pop_size = 8,
    fitness_func = fitness,
    n_parents = 4,
    parent_selector=parent_selector,
    crossover_func=crossover_func,
    log_func=logger,
    initial_genomes = []
)
"""

class HyPaEvo():
    def __init__(self, input_model_genome, n_generations, pop_size, fitness_func, n_parents, parent_selector, crossover_func, log_func, initial_genomes):
        self.input_model_genome = input_model_genome
        self.n_generations = n_generations
        self.pop_size = pop_size
        self.fitness_func = fitness_func
        self.n_parents = n_parents
        self.parent_selector = parent_selector
        self.crossover_func = crossover_func
        self.log_func = log_func
        self.initial_genomes = initial_genomes
    
    def initialize_population(self):
        population = []
        
    # to be continued ...
    
    # def run(self):
    #     population = self.initialize_population()
    #     best_individual = random.choice(population)
        
    #     for gen_idx in range(self.n_generations):
            
    #         self.logger("\n================================================================")
    #         self.prio_logger(f"======================= Generation {gen_idx + 1}/{self.n_generations} =====================")
    #         self.logger("================================================================\n")
            
    #         # Evaluate fitness of population
    #         for i, seqevo_genome in enumerate(population):
    #             self.logger(f"{self.marker_symbol} Evaluating {i+1}/{len(population)} ...\n{seqevo_genome}")
    #             SeqEvoModelChecker.check_model_genome(seqevo_genome)

    #             # get fitness of seqevo_genome
    #             # TODO: if we have a lot of duplicates, the gen_distribution is inaccurate, while no new individual mutate("low")
    #             # by fitting
    #             if seqevo_genome.get_architecture_identifier() not in self.modelcache:
    #                 model_genome = SeqEvoModelGenome.create_with_default_params(seqevo_genome)
    #                 seqevo_genome.fitness = self.fitness_func(model_genome=model_genome, log_func=self.logger)

    #                 # cache fitness 
    #                 self.modelcache[seqevo_genome.get_architecture_identifier()] = seqevo_genome.fitness 
    #                 self.logger(f"=> evaluated fitness: {seqevo_genome.fitness}\n")

    #             # from cache
    #             else:
    #                 seqevo_genome.fitness = self.modelcache[seqevo_genome.get_architecture_identifier()]
    #                 self.logger(f"=> !! fitness from cache: {seqevo_genome.fitness}\n")

    #             # History
    #             self.seqevo_history.write(
    #                 seqevo_genome=seqevo_genome,
    #                 n_generations=gen_idx + 1,
    #             )                           
        
    #         # Rank population
    #         population.sort(key=attrgetter('fitness'), reverse=True)
    #         self.prio_logger(f'{self.marker_symbol} Ranking:\n{self.to_list_str_beauty(population)}\n')

    #         # Adaptive Evolution - EvoTechnique Tracker
    #         # techniques_per_individual = [list(filter(lambda techn: techn.name == individual.created_from, self.techniques)) for individual in population]
    #         # assert all(list(map(lambda t_p_i: len(t_p_i) == 1, techniques_per_individual))), "there are individuals that have not exactly one match with a technique"
    #         benchmark_fitness = best_individual.fitness
    #         for technique in self.techniques:
    #             indis = list(filter(lambda ind: ind.created_from == technique.name, population))
    #             technique.fitnesses.append([indi.fitness for indi in indis])
    #             for indi in indis:
    #                 if indi.fitness > benchmark_fitness:
    #                     technique.improvements.append((gen_idx + 1, indi.fitness))
            
    #         # Update best individual
    #         best_individual_str = None
    #         if population[0].fitness > best_individual.fitness:
    #             best_individual = population[0]
    #             best_individual_str = f"New best individual: {best_individual}"
    #         else:
    #             best_individual_str = f"Old best individual: {best_individual}"
    #         self.prio_logger(f"{self.marker_symbol} {best_individual_str}")
                
    #         # assign next generation
    #         population = self.create_next_generation(population=population, best_individual=best_individual, gen_idx=gen_idx)
    #         assert len(population) == self.pop_size, "got not the pop_size individuals from create_next_generation"

    #     return SeqEvoModelGenome.create_with_default_params(best_individual)
    
