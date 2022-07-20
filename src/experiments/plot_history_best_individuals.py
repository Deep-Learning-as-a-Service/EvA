from optimizer.SeqEvo.SeqEvoHistory import SeqEvoHistory
from matplotlib import pyplot as plt

# Read
seq_hist = SeqEvoHistory(
    path_to_file=f'data/lab/seqevo_history.csv'
)
history_seqevo_genomes = seq_hist.read()
labels = []
plt.figure(figsize=(9,4))
    
x = []
y = []
x2 = []
y2 = []
global_max_fitness = 0
gen_bucket = [[] for i in range(150)]
for gen in history_seqevo_genomes:
    gen_bucket[gen.n_generations - 1].append(gen)

for i, generation in enumerate(gen_bucket[:140]):
    max_fitness = max(max([genome.fitness for genome in generation]), global_max_fitness)
    global_max_fitness = max(max_fitness, global_max_fitness)
    x.append(i+1),
    y.append(max_fitness)
    x2.append(i+1)
    y2.append(max([genome.fitness for genome in generation]))

plt.plot(x, y)
plt.plot(x2, y2)
plt.legend(["Bestes Individuum aller bisherigen Generationen", "Bestes Individuum der momentanen Generation"], loc='lower right')
plt.xlabel("Generation")
plt.ylabel("Fitness")

plt.show()