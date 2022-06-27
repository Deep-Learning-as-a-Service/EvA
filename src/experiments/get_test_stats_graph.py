
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

path = "data/test_data_to_plot"
files = os.listdir(path)
labels = []
plt.figure(figsize=(18,8))
for file in files:
    
    x = []
    y = []
    
    with open (os.path.join("C:/Users/Leander/HPI/BA/eva/data/test_data_to_plot", file), "r") as myfile:
        data = myfile.read().splitlines()
        for line in data:
            generation, timeneeded, acc = [val.strip() for val in line.split(':')]
            x.append(int(generation)),
            y.append(float(acc))
            

    plt.plot(x, ema(y, .8))
    labels.append(file)
plt.legend(labels, loc='upper left')

plt.show()