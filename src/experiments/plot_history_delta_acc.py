from optimizer.SeqEvo.SeqEvoHistory import SeqEvoHistory
from matplotlib import pyplot as plt
import numpy as np

limit = None
# Read
seq_hist = SeqEvoHistory(
    path_to_file=f'data/opportunity/seqevo_history.csv'
)
history_seqevo_genomes = seq_hist.read()
history_seqevo_genomes = list(filter(lambda h_genome: h_genome.n_generations <= 140, history_seqevo_genomes))
print(max([gen.n_generations for gen in history_seqevo_genomes]))
print(min([gen.n_generations for gen in history_seqevo_genomes]))


techniques = list(set(list(map(lambda h_genome: h_genome.created_from, history_seqevo_genomes))))
techniques.remove("random_default")

print(techniques)
techniques_to_german_labels = {
    "finetune_best_individual": "Feinabstimmung \n des besten Individuums",
    "mutate_low": "leichte Mutation",
    "mutate_mid": "mittlere Mutation",
    "mutate_high": "hohe Mutation",
    "mutate_all": "gesamte Mutation",
    "uniform_crossover": "uniforme Kreuzung",
    "middlepoint_crossover": "Mittelpunkt Kreuzung"
}
german_labels_to_techniques = {v: k for k, v in techniques_to_german_labels.items()}
tech_list = list(techniques_to_german_labels.keys())
technique_buckets = {}
for technique in techniques:
    technique_buckets[technique] = list(filter(lambda h_genome: h_genome.created_from == technique, history_seqevo_genomes))

labels = []
technique_boxes = [[] for i in range(7)]

plt.figure(figsize=(18,8))
global_max_fitness = 0
for i, techn in enumerate(tech_list):
    technique_bucket = technique_buckets[techn]
    y = []
    for genome in technique_bucket:
        if not limit:
            y.append(genome.fitness - genome.parent_fitness)
        elif abs(genome.fitness - genome.parent_fitness) < limit:
            y.append(genome.fitness - genome.parent_fitness)



    technique_boxes[i] = y
    labels.append(tech_list)

plt.violinplot(technique_boxes)
plt.xticks(ticks= range(1,8),labels=list(techniques_to_german_labels.values()))
plt.ylabel(ylabel = "Delta Fitness")


plt.show()


