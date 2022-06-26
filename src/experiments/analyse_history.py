from optimizer.SeqEvo.SeqEvoHistory import SeqEvoHistory
from matplotlib import pyplot
import random
from utils import Visualizer

# Read
folder_name = "22-06-22_17-53-38_455253-seqevo_finally-22-06-22_17-53-19_770712"
seq_hist = SeqEvoHistory(
    path_to_file=f'src/saved_experiments/{folder_name}/seqevo_history.csv'
)
history_seqevo_genomes = seq_hist.read()

# Test
print(history_seqevo_genomes[0])
print(history_seqevo_genomes[0].seqevo_genome.layer_list_str())

# Generation Buckets
generation_buckets = []
max_n_generations = history_seqevo_genomes[-1].n_generations
for i in range(max_n_generations):
    generation_buckets.append(list(filter(lambda h_genome: h_genome.n_generations == i+1, history_seqevo_genomes)))

# Technique Buckets
techniques = list(set(list(map(lambda h_genome: h_genome.created_from, history_seqevo_genomes))))
technique_buckets = {}
for technique in techniques:
    technique_buckets[technique] = list(filter(lambda h_genome: h_genome.created_from == technique, history_seqevo_genomes))

# Plots ----------
def accuracy_all_plot():
    points = []
    for i, generation_buck in enumerate(generation_buckets):
        x_axis = i+1
        for h_genome in generation_buck:
            if h_genome.created_from != "initial_models":
                continue
            y_axis = h_genome.fitness
            color = {
                "mutate_all": "green",
                "random_default": "blue",
                "mutate_mid": "yellow",
                "mutate_low": "black",
                "uniform_crossover": "red",
                "middlepoint_crossover": "magenta",
                "mutate_high": "cyan",
                "finetune_best_individual": "#DEB887",
                "initial_models": "#696969"
            }[h_genome.created_from]

            points.append((x_axis, y_axis, color))
        
    Visualizer.multicolor_scatter_plot(points)


def distribution_plot():
    points = []
    y_axis = None
    x_axis = None
    for h_genome in history_seqevo_genomes:
        seqevo_genome = h_genome.seqevo_genome
        for layer in seqevo_genome.layers:
            if layer.__class__.__name__ != "PDenseLayer":
                continue
            
            for param in layer.params:
                if param._key == "units":
                    points.append((h_genome.fitness, param.value, "blue"))
                    break

    Visualizer.multicolor_scatter_plot(points)

distribution_plot()
            



# Idea: Generation x-axis, y axis fitness techniques coloured differnty
# for techni, buck in technique_buckets.items():
#     points = []
#     for i, h_genome in enumerate(buck):
#         points.append((i, h_genome.fitness))
#     show_line_from_points(points)

# Show how good the individuals are, that produced a technique over time

