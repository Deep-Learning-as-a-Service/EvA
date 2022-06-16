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
    
    """
    op
    """
    
