from dataclasses import dataclass

@dataclass
class HistorySeqEvoGenome():
    def __init__(self, seqevo_genome, n_generations, created_unix_timestamp, fitness, created_from, src_file):
        self.seqevo_genome = seqevo_genome
        self.n_generations = n_generations
        self.fitness = fitness
        self.created_from = created_from
        self.created_unix_timestamp = created_unix_timestamp
        self.src_file = src_file
