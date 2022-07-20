import matplotlib.pyplot as plt
import numpy as np


labels = ['Standard Hyperparameter', 'Optuna optimierte Hyperparameter']
CNNLstm = [0.6191, 0.6381]
ResNet = [0.6093, 0.6414]
InnoHAR = [0.6202, 0.6381]
our = [0.6687, 0.6996]

x = np.arange(len(labels))  # the label locations
width = 0.4  # the width of the bars

fig, ax = plt.subplots()
rects1 = ax.bar(x - 3*width/4, CNNLstm, width/2, label='CNNLstm')
rects2 = ax.bar(x - width/4, ResNet, width/2, label='ResNet')
rects3 = ax.bar(x + width/4, InnoHAR, width/2, label='InnoHAR')
rects4 = ax.bar(x + 3*width/4, our, width/2, label='Unser bestes Individuum')



# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Durchschnittliche Genauigkeit')
ax.set_xticks(x, labels)
ax.legend()

ax.bar_label(rects1, padding=3)
ax.bar_label(rects2, padding=3)
ax.bar_label(rects3, padding=3)
ax.bar_label(rects4, padding=3)



ax.set_ylim(bottom=0.4, top = 0.85)


#fig.tight_layout()

plt.show()