

from operator import attrgetter
import random
from model_representation.ModelChecker.SeqEvoModelChecker import SeqEvoModelChecker
from optimizer.SeqEvo.SeqEvoGenome import SeqEvoGenome
from model_representation.ModelGenome.SeqEvoModelGenome import SeqEvoModelGenome
from model_representation.ModelGenome import ModelGenome
import copy
from utils.Tester import Tester
from utils.mutation_helper import get_key_from_prob_dict
from utils.progress_bar import print_progress_bar
from optimizer.SeqEvo.Crosser import Crosser
import time


class SeqEvo():

    def __init__(self, n_generations, pop_size, fitness_func, n_parents, technique_config, parent_selector, log_func, seqevo_history, initial_models, tester=None):
        
        self.n_generations = n_generations
        self.pop_size = pop_size
        self.fitness_func = fitness_func
        self.n_parents = n_parents
        self.technique_config = technique_config
        self.techniques = technique_config.techniques
        self.tester = tester

        self.parent_selector = parent_selector
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
            if i < len(self.initial_models):
                population.append(SeqEvoGenome(self.initial_models[i], created_from="initial_models"))
            else:
                population.append(SeqEvoGenome.create_random())
        return population

    
    def create_next_generation(self, population, best_individual, gen_idx):
        next_generation = []

        # select parents of next generation via selector function
        parents = self.parent_selector(sorted_population = population, n_parents = self.n_parents)

        # adaptive evolution - choose the techniques for the new population
        # for the current optimization_stage, it will use the remaining individuals to choose a random technique of that stage
        relevant_techniques = list(filter(lambda techn: techn.optimization_stage != "none", self.techniques))
        assert self.pop_size >= len(relevant_techniques), "pop_size needs to be at least as big as the number of techniques, to be able to have one individual per technique"
        current_optimization_stage = "macro" if gen_idx+1 < self.technique_config.mid_optimization_start_gen else ("mid" if gen_idx+1 < self.technique_config.micro_optimization_start_gen else "micro")
        n_adaptive_techniques = self.pop_size - len(relevant_techniques)

        technique_name_to_additional_individuals = {t.name:1 for t in relevant_techniques} # start with one individual for each technique
        techniques_of_optimization_stage = list(filter(lambda techn: techn.optimization_stage == current_optimization_stage, self.techniques))
        technique_names_of_optimization_stage = list(map(lambda techn: techn.name, techniques_of_optimization_stage))
        # add individuals to the relevant techniques
        for _ in range(n_adaptive_techniques): 
            technique_name_new_indi = random.choice(technique_names_of_optimization_stage)
            technique_name_to_additional_individuals[technique_name_new_indi] += 1

        # build the next generation
        def add_individuals_of_technique(techn_name, creation_func):
            n_individuals = technique_name_to_additional_individuals[techn_name]
            for _ in range(n_individuals):
                next_generation.append(creation_func())
        
        
        for technique in relevant_techniques:
        
            creation_func = None
            if technique.name == "finetune_best_individual":
                def creation_func():
                    child_best_individual = best_individual.mutate("low")
                    child_best_individual.created_from = "finetune_best_individual"
                    return child_best_individual

            elif technique.name == "middlepoint_crossover":
                def creation_func():
                    pa, ma = random.sample(parents, 2) # pick 2 parent random
                    return Crosser.middlepoint_crossover(pa, ma)
            
            elif technique.name == "uniform_crossover":
                def creation_func():
                    pa, ma = random.sample(parents, 2) # pick 2 parent random
                    return Crosser.uniform_crossover(pa, ma)

            elif technique.name in [f"mutate_{mutation_intensity}" for mutation_intensity in ["low", "mid", "high", "all"]]:
                mutation_intensity = technique.name[7:]
                def creation_func():
                    parent_to_mutate = random.choice(parents)
                    mutated_child = parent_to_mutate.mutate(mutation_intensity)
                    return mutated_child
            elif technique.name == "random":
                def creation_func():
                    return SeqEvoGenome.create_random()
            else:
                raise ValueError("Unknown technique: " + technique.name)

            add_individuals_of_technique(
                techn_name=technique.name, 
                creation_func=creation_func
            )

        return next_generation

    
    def run(self):
        
        population = self.initialize_population()
        best_individual = random.choice(population)
        
        for gen_idx in range(self.n_generations):
            
            self.logger("\n================================================================")
            self.prio_logger(f"======================= Generation {gen_idx + 1}/{self.n_generations} =====================")
            self.logger("================================================================\n")
            starttime = time.time()
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

                # History
                self.seqevo_history.write(
                    seqevo_genome=seqevo_genome,
                    n_generations=gen_idx + 1,
                )                           
        
            # Rank population
            population.sort(key=attrgetter('fitness'), reverse=True)
            self.prio_logger(f'{self.marker_symbol} Ranking:\n{self.to_list_str_beauty(population)}\n')

            # Adaptive Evolution - EvoTechnique Tracker
            # techniques_per_individual = [list(filter(lambda techn: techn.name == individual.created_from, self.techniques)) for individual in population]
            # assert all(list(map(lambda t_p_i: len(t_p_i) == 1, techniques_per_individual))), "there are individuals that have not exactly one match with a technique"
            benchmark_fitness = best_individual.fitness
            for technique in self.techniques:
                indis = list(filter(lambda ind: ind.created_from == technique.name, population))
                technique.fitnesses.append([indi.fitness for indi in indis])
                for indi in indis:
                    if indi.fitness > benchmark_fitness:
                        technique.improvements.append((gen_idx + 1, indi.fitness))
            
            # Update best individual
            best_individual_str = None
            if population[0].fitness > best_individual.fitness:
                best_individual = population[0]
                best_individual_str = f"New best individual: {best_individual}"
            else:
                best_individual_str = f"Old best individual: {best_individual}"
            self.prio_logger(f"{self.marker_symbol} {best_individual_str}")
                
            # assign next generation
            population = self.create_next_generation(population=population, best_individual=best_individual, gen_idx=gen_idx)
            assert len(population) == self.pop_size, "got not the pop_size individuals from create_next_generation"

            if self.tester:
                self.tester.log_test_accuracy(
                    model_genome=SeqEvoModelGenome.create_with_default_params(best_individual), 
                    current_gen_idx=gen_idx,
                    time=time.time() - starttime
                    )

        return SeqEvoModelGenome.create_with_default_params(best_individual)
                    
                