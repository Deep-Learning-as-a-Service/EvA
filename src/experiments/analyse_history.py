from optimizer.SeqEvo.SeqEvoHistory import SeqEvoHistory


folder_name = "22-06-20_17-44-13_192831-TEST"
seq_hist = SeqEvoHistory(
    path_to_file=f'src/saved_experiments/{folder_name}/seqevo_history.csv'
)
history_seqevo_genomes = seq_hist.read()

# Test
print(history_seqevo_genomes[0])
print(history_seqevo_genomes[0].seqevo_genome.layer_list_str())

generation_buckets = []
max_n_generations = history_seqevo_genomes[-1].n_generations

for i in range(max_n_generations):
    generation_buckets.append(list(filter(lambda h_genome: h_genome.n_generations == i+1, history_seqevo_genomes)))

# to be continued ....

