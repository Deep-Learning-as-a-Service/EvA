import time
from optimizer.SeqEvo.SeqEvoHistory import SeqEvoHistory


class AdaptiveSeqEvoHistory(SeqEvoHistory):
    def __init__(self, path_to_file):
        super().__init__(path_to_file=path_to_file)
        self.header = ['created_unix_timestamp', 'n_generation', 'optimization_stage', 'current_fitness_threshold', 'best_seqevo_genome', 'best_seqevo_genome_fitness', 'best_seqevo_genome_technique']
    
    def write(self, n_generation, optimization_stage, current_fitness_threshold, best_seqevo_genome) -> None:

        created_unix_timestamp = time.time()
        
        best_seqevo_genome_fitness = best_seqevo_genome.fitness
        best_seqevo_genome_technique = best_seqevo_genome.created_from
        best_seqevo_genome = best_seqevo_genome.layer_list_str()

        data_row = [created_unix_timestamp, n_generation, optimization_stage, current_fitness_threshold, best_seqevo_genome, best_seqevo_genome_fitness, best_seqevo_genome_technique]
        self._write_row(data_row)
    
    def read(self):
        """
        TODO: could be the visualization
        """ 
        raise Exception("not implemented")

