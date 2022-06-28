
import os

from matplotlib import pyplot as plt
def ema(scalars, weight):  
    last = scalars[0]  
    smoothed = []
    for point in scalars:
        smoothed_val = last * weight + (1 - weight) * point  
        smoothed.append(smoothed_val)                       
        last = smoothed_val                                 
    return smoothed

the_path = "/Users/valentin/github/EvA/src/saved_experiments/22-06-27_00-02-50_848077-seqevo_finally_hermes-22-06-27_00-02-32_116079"

# path = "data/test_data_to_plot"
# files = os.listdir(path)
labels = []
plt.figure(figsize=(18,8))
for file in range(1):
    
    x = []
    y = []
    
    with open (os.path.join(the_path, "tester.txt"), "r") as myfile:
        data = myfile.read().splitlines()
        for line in data:
            generation, timeneeded, acc = [val.strip() for val in line.split(':')]
            x.append(int(generation)),
            y.append(float(acc))
            

    plt.plot(x, y)
    labels.append(file)
plt.legend(labels, loc='upper left')

plt.show()
# plt.savefig(os.path.join(the_path, "test_data_to_plot.png"))