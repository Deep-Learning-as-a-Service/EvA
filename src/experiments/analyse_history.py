from optimizer.SeqEvo.SeqEvoHistory import SeqEvoHistory
from matplotlib import pyplot
import random

# Read
folder_name = "22-06-22_17-53-38_455253-seqevo_finally-22-06-22_17-53-19_770712"
seq_hist = SeqEvoHistory(
    path_to_file=f'src/saved_experiments/{folder_name}/seqevo_history.csv'
)
history_seqevo_genomes = seq_hist.read()

# Test
print(history_seqevo_genomes[0])
print(history_seqevo_genomes[0].seqevo_genome.layer_list_str())

# Funcs
def points_to_two_lists(points):
    xPoints = list(map(lambda point: point[0], points))
    yPoints = list(map(lambda point: point[1], points))
    return xPoints, yPoints

def show_line_from_points(points):
    """
    show_point_plot([(1, 1), (1.1, 3), (7, 8), (0.9, 4)])
    """
    xPoints, yPoints = points_to_two_lists(points)
    pyplot.plot(xPoints, yPoints)
    pyplot.show()

def show_scatter_plot(points):

    # Set the figure size in inches
    pyplot.figure(figsize=(10,6))

    xPoints, yPoints = points_to_two_lists(points)
    pyplot.scatter(xPoints, yPoints, label = "label_name", alpha=0.5)

    # Set x and y axes labels
    pyplot.xlabel('X Values')
    pyplot.ylabel('Y Values')

    pyplot.title('Scatter Title')
    pyplot.legend()
    pyplot.show()

def multicolor_scatter_plot(color_points):
    xPoints = list(map(lambda point: point[0], color_points))
    yPoints = list(map(lambda point: point[1], color_points))
    colours = list(map(lambda point: point[2], color_points))
    
    pyplot.figure(figsize=(10,6))

    pyplot.scatter(xPoints, yPoints, c = colours, label = "label_name", alpha=0.05)

    # Set x and y axes labels
    pyplot.xlabel('X Values')
    pyplot.ylabel('Y Values')

    pyplot.title('Scatter Title')
    pyplot.legend()
    pyplot.show()


# Test
# show_scatter_plot([(5, 5), (5, 5), (6, 7), (1, 2), (5.1, 5), (5, 5.1)])
# show_line_from_points([(1, 1), (1.1, 3), (7, 8), (0.9, 4)])
# show_scatter_plot([(6, 7), (1, 2), (5.1, 5), (5, 5.1)])
# multicolor_scatter_plot([(6, 7, "green"), (1, 2, "red"), (5.1, 5, "red"), (5, 5.1, "blue")])


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
        
    multicolor_scatter_plot(points)


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

    multicolor_scatter_plot(points)

distribution_plot()
            



# Idea: Generation x-axis, y axis fitness techniques coloured differnty
# for techni, buck in technique_buckets.items():
#     points = []
#     for i, h_genome in enumerate(buck):
#         points.append((i, h_genome.fitness))
#     show_line_from_points(points)

# Show how good the individuals are, that produced a technique over time

