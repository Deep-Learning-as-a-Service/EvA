from optimizer.SeqEvo.SeqEvoHistory import SeqEvoHistory
from matplotlib import pyplot
import random

# Read
folder_name = "22-06-20_17-44-13_192831-TEST"
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

# Test
# show_scatter_plot([(5, 5), (5, 5), (6, 7), (1, 2), (5.1, 5), (5, 5.1)])
# show_line_from_points([(1, 1), (1.1, 3), (7, 8), (0.9, 4)])
# show_scatter_plot([(6, 7), (1, 2), (5.1, 5), (5, 5.1)])


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


# Idea: Generation x-axis, y axis fitness techniques coloured differnty

# Show how good the individuals are, that produced a technique over time
for techni, buck in technique_buckets.items():
    points = []
    for i, h_genome in enumerate(buck):
        points.append((i, h_genome.fitness))
    show_line_from_points(points)


