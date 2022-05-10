import random

class Selector():
    
    @staticmethod
    def select_from_fitness_probability(sorted_population, n_parents):
        """
        returns n_parents, their probability of being chosen is weighted linear via their fitness value
        """
        sorted_pop = sorted_population
        chosen_genomes = []
        
        # choose n_parents
        for _ in range(n_parents):
            rand_number = random.random()
            genomes_with_prob_range = []
            chosen_genome = None
            start_value = 0
            
            # get sum of population fitness
            total_fitness = sum([genome.fitness for genome in sorted_pop])
            
            # get genomes_with_prob_range of type [(genome_01, probability_start_01), ...]
            for genome in sorted_pop:
                genomes_with_prob_range.append((genome, start_value))
                start_value += (genome.fitness / total_fitness)

            # loop over genomes_with_prob_range
            for idx in range(len(genomes_with_prob_range)):
                
                # skip first element
                if idx == 0: 
                    continue
                
                # if last element reached, last element from genomes_with_prob_range is chosen
                elif idx == len(genomes_with_prob_range) - 1:
                    chosen_genome = genomes_with_prob_range[-1][0]
                    break
                
                # if start_range is smaller than probability_start, continue
                elif genomes_with_prob_range[idx][1] < rand_number:
                    continue
                
                # if start_range is higher than probability_start, take genome before the current one
                else:
                    chosen_genome = genomes_with_prob_range[idx-1][0]
                    break
                
            # append chosen genome to chosen genome list and remove chosen genome from list
            chosen_genomes.append(chosen_genome)
            sorted_pop.remove(chosen_genome)
            
        return chosen_genomes
