from optimizer.SeqEvo.SeqEvo import SeqEvo
from optimizer.SeqEvo.Selector import Selector
import random


""""
concept

- generation_size = 5
- n_parents = 2
- choose best_parents always
- always focus on one variation strength: micro | mid | macro

- wenn mico 5 Generationen ohne Erfolg, dann gibt es da nichts mehr zu holen
- der aktuelle Gütethreshold unabhängig von global muss in den Generationen muss absinken - wir entwickeln pyramidal

- we start with 0.5 threshold
- first gen only macro if we go above the threshold, a specific genome was found, that is good, and needs some finetuning?
- macro until we get over the threshold? - high threshold
- mid either 3 Generations or threshold
- micro either 3 Generations or threshold - again 3 

Future:
- make a bigger pool? has 4 generations time to come up with good individuals?
- than the best go to the next development stage... pools are passed?




immer macro
wenn Verbesserung des globalen Optimas Übergang in mid



"""

class AdaptiveSeqEvo(SeqEvo):

    def __init__(self, n_generations, fitness_func, technique_config, seqevo_history, log_func):
        pop_size = 5
        n_parents = 2
        parent_selector = Selector.select_best
        initial_models = []
        
        self.basic_fitness_threshold = 0.6
        self.current_fitness_threshold = self.basic_fitness_threshold

        self.basic_optimization_stage = "macro"
        self.current_optimization_stage = self.basic_optimization_stage

        self.generations_since_threshold_improvement = 0
        self.generations_limit_no_improvement = 3

        super().__init__(n_generations, pop_size, fitness_func, n_parents, technique_config, parent_selector, log_func, seqevo_history, initial_models, tester=None)
    
    def go_to_next_optimization_stage(self, current_stage):
        if current_stage == "macro": return "mid"
        elif current_stage == "mid": return "micro"
        else: return "macro"
    
    def go_to_previous_optimization_stage(self, current_stage):
        if current_stage == "macro": return "micro"
        elif current_stage == "mid": return "macro"
        else: return "mid"

    def create_next_generation(self, population, best_individual, gen_idx):
        next_generation = []

        # select parents of next generation via selector function
        parents = self.parent_selector(sorted_population = population, n_parents = self.n_parents) if self.current_optimization_stage != "macro" else population

        # Change optimization stage
        best_individual_last_population = population[:1][0]
        best_individual_fitness_last_population = best_individual_last_population.fitness
        difference_to_threshold = self.current_fitness_threshold - best_individual_fitness_last_population
        self.current_fitness_threshold = best_individual_fitness_last_population

        old_optimization_stage = self.current_optimization_stage

        # in the macro stage make the threshold smaller over time
        if self.current_optimization_stage == "macro" and self.current_fitness_threshold > self.basic_fitness_threshold:
            self.current_fitness_threshold -= 0.005

        # better individual, go to next stage
        if difference_to_threshold < 0 and self.current_optimization_stage != "micro":
            self.current_optimization_stage = self.go_to_next_optimization_stage(self.current_optimization_stage)
        
        # count how long already in optimization stage
        if self.current_optimization_stage != old_optimization_stage:
            self.generations_since_threshold_improvement = 0
        else:
            self.generations_since_threshold_improvement += 1
        
        # no better individual, go back to stage
        if self.generations_since_threshold_improvement > self.generations_limit_no_improvement and self.current_optimization_stage != "macro":
            self.current_optimization_stage = self.go_to_previous_optimization_stage(self.current_optimization_stage)

        # adaptive evolution - choose the techniques for the new population
        # for the current optimization_stage, it will use the remaining individuals to choose a random technique of that stage
        # development_techniques = list(filter(lambda techn: techn.optimization_stage != "none", self.techniques))
        techniques_of_optimization_stage = list(filter(lambda techn: techn.optimization_stage == self.current_optimization_stage, self.techniques))
        technique_names_of_optimization_stage = list(map(lambda techn: techn.name, techniques_of_optimization_stage))


        technique_name_to_number_of_individuals = {"finetune_best_individual": 1}
        for _ in range(self.pop_size - 1): 
            technique_name_new_indi = random.choice(technique_names_of_optimization_stage)
            if technique_name_new_indi in technique_name_to_number_of_individuals:
                technique_name_to_number_of_individuals[technique_name_new_indi] += 1
            else:
                technique_name_to_number_of_individuals[technique_name_new_indi] = 1

        # TODO: refactor! needs seperate method

        # build the next generation
        def add_individuals_of_technique(techn_name, creation_func):
            if techn_name not in technique_name_to_number_of_individuals: return
            n_individuals = technique_name_to_number_of_individuals[techn_name]
            for _ in range(n_individuals):
                next_generation.append(creation_func())


        for technique in self.techniques:

            creation_func = None
            if technique.name == "finetune_best_individual":
                def creation_func():
                    child_best_individual = best_individual.mutate("low")
                    child_best_individual.created_from = "finetune_best_individual"
                    return child_best_individual

            elif technique.name == "middlepoint_crossover":
                def creation_func():
                    pa, ma = parents[0], parents[1] if self.current_optimization_stage != "macro" else random.sample(parents, 2)
                    return Crosser.middlepoint_crossover(pa, ma)
            
            elif technique.name == "uniform_crossover":
                def creation_func():
                    pa, ma = parents[0], parents[1] if self.current_optimization_stage != "macro" else random.sample(parents, 2)
                    return Crosser.uniform_crossover(pa, ma)

            elif technique.name in [f"mutate_{mutation_intensity}" for mutation_intensity in ["low", "mid", "high", "all"]]:
                mutation_intensity = technique.name[7:]
                def creation_func():
                    parent_to_mutate = best_individual_last_population if self.current_optimization_stage != "macro" else random.choice(parents)
                    mutated_child = parent_to_mutate.mutate(mutation_intensity)
                    return mutated_child
            elif technique.name == "random":
                def creation_func():
                    return SeqEvoGenome.create_random()
            else:
                pass
                # raise ValueError("Unknown technique: " + technique.name)

            add_individuals_of_technique(
                techn_name=technique.name, 
                creation_func=creation_func
            )

        return next_generation