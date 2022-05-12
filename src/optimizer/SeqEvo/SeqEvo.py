

from operator import attrgetter
import random
from optimizer.SeqEvo.SeqEvoGenome import SeqEvoGenome
from model_representation.ModelGenome.SeqEvoModelGenome import SeqEvoModelGenome
from model_representation.ModelGenome import ModelGenome
import copy
from utils.mutation_helper import get_mutation_probability

class SeqEvo():
    intensitiy_to_genome_mutation = {
        "low" : {
            "add_node_random" : 0.01,
            "remove_node_random": 0.01,
            "none": 0.98
        },
        "mid" : {
            "add_node_random" : 0.1,
            "remove_node_random": 0.1,
            "none": 0.8
        },
        "high" : {
            "add_node_random" : 0.25,
            "remove_node_random": 0.25,
            "none": 0.5
        },
        "all" : {
            "add_node_random" : 0.5,
            "remove_node_random": 0.5,
            "none": 0.0
        } 
    }
    def __init__(self, n_generations, pop_size, fitness_func, n_parents, generation_distribution, parent_selector, crossover_func):
        
        assert sum(generation_distribution.values()) == 1.0, "sum of generation distribution must be 1"
        self.n_generations = n_generations
        self.pop_size = pop_size
        self.fitness_func = fitness_func
        self.n_parents = n_parents
        self.generation_distribution = generation_distribution

        self.parent_selector = parent_selector
        self.crossover_func = crossover_func
        
    def initialize_population(self):
        population = []
        for _ in range(self.pop_size):
            population.append(SeqEvoGenome.create_random())
        return population
            
    def evaluate_current_population(self, population):
        """
        set the model_genome.fitness values of the population
        """
        for seqevo_genome in population:
            model_genome = SeqEvoModelGenome.create_with_default_params(seqevo_genome)
            seqevo_genome.fitness = self.fitness_func(model_genome)
            
    def pick_two_parents_random(self, parents):
        return random.sample(parents, 2)
    
    def run(self):
        
        ####################################################################################################
        #                                   still important TODO:                                          #
        #                                                                                                  #
        # - (DONE) mutate layer itself with a certain probability (given via intensity)                    #
        # - (DONE) define layer params mutation probabilities for a given layer mutation intensity         #
        # - add + remove layers / ModelNodes within ModelGenomes with a certain probability                #
        # - make_compatible() is still unfinished, need for more assertions (at a certain amount           #
        #       of Conv1Ds, one dimension folds to 0, which throws an error at runtime)                    #
        # - on uneven pop_size there is an issue with the calculation of n_mutation_childs and             #
        #       n_crossover_childs                                                                         #
        ####################################################################################################
        
        population = self.initialize_population()
        best_individual = random.choice(population)
        
        for gen_idx in range(self.n_generations):
            
            print("================================================================")
            print("========================= Generation " + str(gen_idx) + " =========================")
            print("================================================================")
            
            # evaluate current population inplace
            self.evaluate_current_population(population = population)
            
            # sorts population via fitness inplace, highest ranked first 
            population.sort(key=attrgetter('fitness'), reverse=True)
            
            # update best individual
            if(population[0].fitness > best_individual.fitness):
                best_individual = population[0]
                
            # select parents of next generation via selector function
            parents = self.parent_selector(sorted_population = population, n_parents = self.n_parents)
            next_generation = []
            
            # get childs from crossover fucntion
            n_crossover_childs = round(self.generation_distribution["crossover"] * self.pop_size)
            for _ in range(n_crossover_childs):
                pa, ma = self.pick_two_parents_random(parents)
                next_generation.append(self.crossover_func(pa, ma))
            
            # get childs from mutations
            for mutation_intensity in ["low", "mid", "high", "all"]:
                n_mutation_childs = round(self.generation_distribution["mutate_" + mutation_intensity] * self.pop_size)
                for _ in range(n_mutation_childs):
                    # add/remove nodes
                    parent_to_mutate = copy.deepcopy(random.choice(parents))
                    prob_dict = SeqEvo.intensitiy_to_genome_mutation[mutation_intensity]
                    genome_mutation = get_mutation_probability(prob_dict)

                    if genome_mutation == "add_node_random":
                        parent_to_mutate.add_node_random()
                    elif genome_mutation == "remove_node_random":
                        parent_to_mutate.remove_node_random()
                    
                    next_generation.append(parent_to_mutate.mutate(mutation_intensity))
            
            # assign next generation
            print("BEST FITNESS IN THIS GENERATION: " + str(best_individual.fitness))
            print("LAYERS OF THE GENOME: " + str(best_individual.layers))
            print("NEXT GENERATION LENGTH: " + str(len(next_generation)))
            population = next_generation
        
        return SeqEvoModelGenome.create_with_default_params(best_individual)
                    
                