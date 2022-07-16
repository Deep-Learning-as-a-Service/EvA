from optimizer.SeqEvo.SeqEvoHistory import SeqEvoHistory

# Read
seq_hist = SeqEvoHistory(
    path_to_file=f'data/seqevo_history.csv'
)
history_seqevo_genomes = seq_hist.read()
labels = []
    

global_max_fitness = 0
global_best_gen = None
gen_bucket = [[] for i in range(150)]
for gen in history_seqevo_genomes:
    gen_bucket[gen.n_generations - 1].append(gen)

for i, generation in enumerate(gen_bucket):
    if max([genome.fitness for genome in generation]) > global_max_fitness:
        best_genome = [genome for genome in generation if genome.fitness == max([genome.fitness for genome in generation])][0]
        print(best_genome.created_from)
        print(best_genome.n_generations)
        if global_best_gen:
            print((best_genome.fitness - global_best_gen.fitness) * 100)
        print(best_genome.seqevo_genome)
        global_best_gen = best_genome

    max_fitness = max(max([genome.fitness for genome in generation]), global_max_fitness)

    global_max_fitness = max(max_fitness, global_max_fitness)

