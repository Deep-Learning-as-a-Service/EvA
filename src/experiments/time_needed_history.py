from optimizer.SeqEvo.SeqEvoHistory import SeqEvoHistory

# Read
seq_hist = SeqEvoHistory(
    path_to_file=f'data/opportunity/seqevo_history.csv'
)
history_seqevo_genomes = seq_hist.read()
labels = []
    

global_max_fitness = 0
global_best_gen = None
gen_bucket = [[] for i in range(140)]
for gen in history_seqevo_genomes:
    if gen.n_generations > len(gen_bucket):
        break
    gen_bucket[gen.n_generations - 1].append(gen)

history_seqevo_genomes = [s for s in history_seqevo_genomes if s.n_generations <=140]
print(max([s.n_generations for s in history_seqevo_genomes]))
print(min([s.n_generations for s in history_seqevo_genomes]))
ma = max([s.created_unix_timestamp for s in history_seqevo_genomes])
mi = min([s.created_unix_timestamp for s in history_seqevo_genomes])
total = (ma - mi) / (60)
print(total)

