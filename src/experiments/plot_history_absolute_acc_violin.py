from optimizer.SeqEvo.SeqEvoHistory import SeqEvoHistory
from matplotlib import pyplot as plt
import numpy as np

limit = 0.5
# Read
seq_hist = SeqEvoHistory(
    path_to_file=f'data/seqevo_history.csv'
)
history_seqevo_genomes = seq_hist.read()

techniques = list(set(list(map(lambda h_genome: h_genome.created_from, history_seqevo_genomes))))
technique_buckets = {}
for technique in techniques:
    technique_buckets[technique] = list(filter(lambda h_genome: h_genome.created_from == technique, history_seqevo_genomes))

labels = []
technique_boxes = [[] for i in range(8)]

plt.figure(figsize=(18,8))
global_max_fitness = 0
for i, technique_bucket in enumerate(technique_buckets.values()):
    y = []
    for genome in technique_bucket:
        if not limit:
            y.append(genome.fitness)
        elif genome.fitness > limit:
            y.append(genome.fitness)

    technique_boxes[i] = y
    labels.append(list(technique_buckets.keys())[i])

plt.violinplot(technique_boxes)
plt.xticks(ticks= range(1,9),labels=labels)



plt.show()


