from optimizer.SeqEvo.SeqEvoHistory import SeqEvoHistory
from matplotlib import pyplot
import random
from utils.Visualizer import Visualizer

# Read
folder_name = "22-06-22_17-53-38_455253-seqevo_finally-22-06-22_17-53-19_770712"
folder_name = "22-06-27_00-02-50_848077-seqevo_finally_hermes-22-06-27_00-02-32_116079"
seq_hist = SeqEvoHistory(
    path_to_file=f'src/saved_experiments/{folder_name}/seqevo_history.csv'
)
history_seqevo_genomes = seq_hist.read()

def general_analysis():
    print("num of genomes:", len(history_seqevo_genomes))
    print("last generation:", history_seqevo_genomes[-1].n_generations)
    print("average length of genomes:", sum([len(genome.seqevo_genome.layers) for genome in history_seqevo_genomes]) / len(history_seqevo_genomes))

def mean_median_min_max(numbers):
    numbers = sorted(numbers)
    return (
        sum(numbers) / len(numbers),
        numbers[int(len(numbers) / 2)],
        numbers[0],
        numbers[-1]
    )

def print_mean_median_min_max(numbers):
    mean, median, min_, max_ = mean_median_min_max(numbers)
    print("mean:", mean, "median:", median, "min:", min_, "max:", max_)


def print_bucket_analysis(bucket):

    # generations
    print("generations:")
    generations = list(map(lambda x: x.n_generations, bucket))
    print_mean_median_min_max(generations)
    print("\n")

    # length
    print("length:")
    lengths = list(map(lambda x: len(x.seqevo_genome.layers), bucket))
    print_mean_median_min_max(lengths)
    print("\n")

    # accuracy
    print("accuracy:")
    accuracies = list(map(lambda x: x.fitness, bucket))
    print_mean_median_min_max(accuracies)
    print("\n")

    # delta accuracy
    print("delta accuracy:")
    accuracies = list(map(lambda x: x.fitness - x.parent_fitness, bucket))
    print_mean_median_min_max(accuracies)
    print("\n")

    # layers
    for layi in ["PDenseLayer", "PConv2DLayer", "PConv1DLayer", "PLstmLayer"]:
        n_layer = 0
        for h_genome in bucket:
            seqevo_genome = h_genome.seqevo_genome
            for layer in seqevo_genome.layers:
                if layer.__class__.__name__ == layi:
                    n_layer += 1
        print("number of ", layi, ":", n_layer)

def print_buckets_analysis(title, buckets_dict):
    print("BUCKET ANALYSIS ", title, " ----------------------------------------------------")

    for bucket_name, genomes in buckets_dict.items():
        print(f"{bucket_name}: (len: {len(genomes)}) -------------------------------")
        print_bucket_analysis(genomes)
        print("\n")

# Test
print(history_seqevo_genomes[0])
print(history_seqevo_genomes[0].seqevo_genome.layer_list_str())

# Generation Buckets
generation_buckets = []
max_n_generations = history_seqevo_genomes[-1].n_generations
for i in range(max_n_generations):
    generation_buckets.append(list(filter(lambda h_genome: h_genome.n_generations == i+1, history_seqevo_genomes)))

generation_buckets_dict = {i: genomes for i, genomes in enumerate(generation_buckets)}

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
            # if h_genome.created_from == "initial_models":
            #     continue
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

            points.append((x_axis, y_axis, color, 0.8))
        
    Visualizer.scatter_plot(points)

def delta_accuracy_technique_plot():
    points = []
    for i, generation_buck in enumerate(generation_buckets):
        x_axis = i+1
        for h_genome in generation_buck:
            if h_genome.created_from not in ["finetune_best_individual"]:
                continue
            y_axis = h_genome.fitness - h_genome.parent_fitness
            color = {
                "mutate_all": "green",
                "random_default": "blue",
                "mutate_mid": "yellow",
                "mutate_low": "black",
                "uniform_crossover": "red",
                "middlepoint_crossover": "orange",
                "mutate_high": "cyan",
                "finetune_best_individual": "#DEB887", # holzfarbe
                "initial_models": "#696969" # grey
            }[h_genome.created_from]

            points.append((x_axis, y_axis, color, 0.8))
        
    Visualizer.scatter_plot(points)
    


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
                    points.append((h_genome.fitness, param.value, "blue", 0.3))
                    break

    Visualizer.scatter_plot(points)

# distribution_plot()
# accuracy_all_plot()
# delta_accuracy_technique_plot()

general_analysis()
# print_buckets_analysis("GENERATION BUCKETS", generation_buckets_dict)
# print_buckets_analysis("TECHNIQUE BUCKETS", technique_buckets)
print_buckets_analysis("ALL", {"all": history_seqevo_genomes})          



# Idea: Generation x-axis, y axis fitness techniques coloured differnty
# for techni, buck in technique_buckets.items():
#     points = []
#     for i, h_genome in enumerate(buck):
#         points.append((i, h_genome.fitness))
#     show_line_from_points(points)

# Show how good the individuals are, that produced a technique over time

