from SeqEvo.HistorySeqEvoGenome import HistorySeqEvoGenome
import time
import csv
import os

class SeqEvoHistory:
    def __init__(self, path_to_file):
        self.path_to_file = path_to_file
        self.header = ['created_unix_timestamp', 'layer_list', 'fitness', 'created_from', 'n_generation']
        self.file_exists = os.path.isfile(self.path_to_file)
    
    def _write_row(self, data_row) -> None:
        assert len(data_row) == len(self.header)
        
        if not self.file_exists:
            with open(self.path_to_file, 'a+', encoding='UTF8') as f:
                writer = csv.writer(f)
                writer.writerow(self.header)

        with open(self.path_to_file, 'a+', encoding='UTF8') as f:
            writer = csv.writer(f)
            writer.writerow(data_row)
    
    def write(self, seq_evo_genome, n_generation) -> None:
        created_unix_timestamp = time.time()
        layer_list = seq_evo_genome.layer_list_str()
        fitness = seq_evo_genome.fitness
        created_from = seq_evo_genome.created_from

        data_row = [created_unix_timestamp, layer_list, fitness, created_from, n_generation]
        self._write_row(data_row)


    def read(self) -> 'list[HistorySeqEvoGenome]':
        """
        TODO
        """
        pass