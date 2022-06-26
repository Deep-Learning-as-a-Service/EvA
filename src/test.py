import numpy as np
import matplotlib.pylab as plt

x = [1, 2, 3]
y = [3, 1, 2]
colors = ["red", "blue", "green"]
alphas = [0.5, 1, 0.3]
labels = ["cool", "semi", "lol"]

plt.scatter(x, y, c=colors, label=labels, alpha=alphas)
plt.legend()
plt.show()
