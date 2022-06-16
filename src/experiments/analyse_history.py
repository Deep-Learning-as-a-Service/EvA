from optimizer.SeqEvo.SeqEvoHistory import SeqEvoHistory

seq_hist = SeqEvoHistory(
    path_to_file='src/saved_experiments/22-06-01_18-26-25_942616-artemis_kfold_10epoch-22-06-01_18-26-08_058027/seqevo_history.csv'
)
history_seqevo_genomes = seq_hist.read()

print(history_seqevo_genomes[0])
print(history_seqevo_genomes[0].seqevo_genome.layer_list_str())