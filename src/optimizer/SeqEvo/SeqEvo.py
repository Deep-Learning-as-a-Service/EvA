

from operator import attrgetter
import random
from model_representation.ModelChecker.SeqEvoModelChecker import SeqEvoModelChecker
from optimizer.SeqEvo.SeqEvoGenome import SeqEvoGenome
from model_representation.ModelGenome.SeqEvoModelGenome import SeqEvoModelGenome
from model_representation.ModelGenome import ModelGenome
import copy
from utils.mutation_helper import get_key_from_prob_dict
from utils.progress_bar import print_progress_bar


class SeqEvo():

    def __init__(self, n_generations, pop_size, fitness_func, n_parents, generation_distribution, parent_selector, crossover_func, log_func, seqevo_history, initial_models):
        
        assert sum(generation_distribution.values()) == pop_size, "sum of generation distribution must be equal to population size"
        self.n_generations = n_generations
        self.pop_size = pop_size
        self.fitness_func = fitness_func
        self.n_parents = n_parents
        self.generation_distribution = generation_distribution

        self.parent_selector = parent_selector
        self.crossover_func = crossover_func
        self.initial_models = initial_models
        self.modelcache = {}

        # Logging
        self.seqevo_history = seqevo_history
        self.logger = log_func
        self.prio_logger = lambda *args, **kwargs: self.logger(*args, prio=True, **kwargs)
        progress_bar = lambda prefix, suffix, progress, total: print_progress_bar(progress, total, prefix = prefix, suffix = suffix, length = 30, log_func = self.logger)
        self.progress_bar_fitting = lambda prefix, progress, total: progress_bar(prefix, '- ' + str(progress) + '/' + str(total) + ' fitted', progress, total)
        self.to_list_str_beauty = lambda li: '[\n\t' + ',\n\n\t'.join(list(map(str, li))) + '\n]'
        self.marker_symbol = '***'
        
    def initialize_population(self):
        population = []
        for i in range(self.pop_size):
            if(i < len(self.initial_models)):
                population.append(SeqEvoGenome(self.initial_models[i], created_from="initial_models"))
            else:
                population.append(SeqEvoGenome.create_random())
        return population
            
    def pick_two_parents_random(self, parents):
        return random.sample(parents, 2)
    
    def create_next_generation(self, population, best_individual):
        next_generation = []

        # select parents of next generation via selector function
        parents = self.parent_selector(sorted_population = population, n_parents = self.n_parents)

        # get childs from best individual mutation
        n_finetuned_childs = self.generation_distribution["finetune_best_individual"]
        for _ in range(n_finetuned_childs):
            child_best_individual = copy.deepcopy(best_individual)
            child_best_individual.mutate("low")
            child_best_individual.fitness = None
            child_best_individual.created_from = "finetuned_best_individual"
            next_generation.append(child_best_individual)

        # get childs from crossover fucntion
        n_crossover_childs = self.generation_distribution["crossover"]
        for _ in range(n_crossover_childs):
            pa, ma = self.pick_two_parents_random(parents)
            next_generation.append(self.crossover_func(pa, ma))
        
        # get childs from mutations
        for mutation_intensity in ["low", "mid", "high"]:
            n_mutation_childs = self.generation_distribution["mutate_" + mutation_intensity]
            for _ in range(n_mutation_childs):
                
                # copy random parent and mutate it with given intensity
                parent_to_mutate = copy.deepcopy(random.choice(parents))

                mutated_child = parent_to_mutate.mutate(mutation_intensity)
                mutated_child.fitness = None
                mutated_child.created_from = "mutate_" + mutation_intensity
                next_generation.append(mutated_child)
        
        # get random individuals for "all" intensity
        n_mutation_childs_all = self.generation_distribution["mutate_all"]
        for _ in range(n_mutation_childs_all):

                mutated_child = SeqEvoGenome.create_random()
                mutated_child.created_from = "mutate_all" 
                next_generation.append(mutated_child)

        return next_generation
    
    def run(self):
        
        population = self.initialize_population()
        best_individual = random.choice(population)
        
        for gen_idx in range(self.n_generations):
            
            self.logger("\n================================================================")
            self.prio_logger(f"======================= Generation {gen_idx + 1}/{self.n_generations} =====================")
            self.logger("================================================================\n")
            
            # Evaluate fitness of population
            for i, seqevo_genome in enumerate(population):
                self.logger(f"{self.marker_symbol} Evaluating {i+1}/{len(population)} ...\n{seqevo_genome}")
                SeqEvoModelChecker.check_model_genome(seqevo_genome)

                # get fitness of seqevo_genome
                # TODO: if we have a lot of duplicates, the gen_distribution is inaccurate, while no new individual mutate("low")
                # by fitting
                if seqevo_genome.get_architecture_identifier() not in self.modelcache:
                    model_genome = SeqEvoModelGenome.create_with_default_params(seqevo_genome)
                    seqevo_genome.fitness = self.fitness_func(model_genome=model_genome, log_func=self.logger)

                    # cache fitness 
                    self.modelcache[seqevo_genome.get_architecture_identifier()] = seqevo_genome.fitness 
                    self.logger(f"=> evaluated fitness: {seqevo_genome.fitness}\n")

                # from cache
                else:
                    seqevo_genome.fitness = self.modelcache[seqevo_genome.get_architecture_identifier()]
                    self.logger(f"=> !! fitness from cache: {seqevo_genome.fitness}\n")

                self.seqevo_history.write(
                    seqevo_genome=seqevo_genome,
                    n_generations=gen_idx + 1,
                )                           
        
            # Rank population
            population.sort(key=attrgetter('fitness'), reverse=True)
            self.prio_logger(f'{self.marker_symbol} Ranking:\n{self.to_list_str_beauty(population)}\n')
            
            # Update best individual
            best_individual_str = None
            if population[0].fitness > best_individual.fitness:
                best_individual = population[0]
                best_individual_str = f"New best individual: {best_individual}"
            else:
                best_individual_str = f"Old best individual: {best_individual}"
            self.prio_logger(f"{self.marker_symbol} {best_individual_str}")
                
            # assign next generation
            population = self.create_next_generation(population=population, best_individual=best_individual)
        
        return SeqEvoModelGenome.create_with_default_params(best_individual)
                    
                