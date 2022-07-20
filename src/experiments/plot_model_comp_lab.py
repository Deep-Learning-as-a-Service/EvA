import matplotlib.pyplot as plt
import numpy as np


labels = ['Standard Hyperparameter', 'Optuna optimierte Hyperparameter']
CNNLstm = [0.3822, 0.4509]
ResNet = [0.4082, 0.4496]
InnoHAR = [0.4325, 0.4325]
our = [0.4777, 0.5073]

x = np.arange(len(labels))  # the label locations
width = 0.4  # the width of the bars


fig, ax = plt.subplots()
fig.figsize =(9,4)

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



ax.set_ylim(bottom=0.3, top = 0.6)


#fig.tight_layout()

plt.show()