import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib import pyplot

# Data
csv_path = "tmp/adaptive_evolution.csv"
output_file_path = "tmp/visualization.png"
df = pd.read_csv(csv_path)

subset_frame = False
if subset_frame:
    df = df[20:90]


# Line: current_fitness_threshold, best_individual_fitness
sns.set_style("white")
plt.figure(figsize=(12, 10))
plt.xlabel('epoch', fontsize=18)
plt.title('Adaptive Evolution', fontsize=22)

sns.lineplot(
    data=df[["current_fitness_threshold", "best_individual_fitness"]])


# Scatter: optimization_stage
color_dict = {
    "macro": "black",
    "mid": "blue",
    "micro": "red"
}

df_to_tuples = lambda d: list(d.itertuples(index=False, name=None))
optimization_tuples = df_to_tuples(df[["n_generation", "optimization_stage"]])

x_points = list(map(lambda tup: tup[0], optimization_tuples))
y_points = [0.62 for _ in range(len(x_points))]
alphas = 1
colours = [color_dict[o_tup[1]] for o_tup in optimization_tuples]
pyplot.scatter(x_points, y_points, c = colours, label = "label_name", alpha=alphas)


plt.savefig(output_file_path)