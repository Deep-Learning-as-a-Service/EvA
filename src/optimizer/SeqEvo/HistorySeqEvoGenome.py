class HistorySeqEvoGenome():
    def __init__(self, seqevo_genome, n_generation, created_unix_timestamp, src_file):
        self.seqevo_genome = seqevo_genome
        self.n_generation = n_generation
        self.created_unix_timestamp = created_unix_timestamp
        self.src_file = src_file
