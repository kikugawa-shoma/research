import matplotlib.pyplot as plt
from statistics import mean, stdev
import numpy as np

root_dir = "C:\\Users\\ktmks\\Documents\\research\\tmp\\results\\"

filenames = [
    "svm_raw",
    "svm_raw_confusion_mat_classified",
    ]

n = len(filenames)
y = [0]*n
err = [0]*n
x = [i for i in range(n)]
width = [0.6]*n
tick_label = ["only\nnormalized","conf-mat\nclassified"]
y_tick = [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]

for i in range(n):
    f = open(root_dir + filenames[i] + ".txt","r")
    acc = list(map(float,f.readline().split()))
    print(acc)
    y[i] = mean(acc)
    print(mean(acc))
    err[i] = stdev(acc)

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.bar(x,y,width=width,yerr=err,capsize=10,tick_label=tick_label)
ax.set_yticks(y_tick)
ax.set_yticklabels(y_tick, fontsize = 15)
ax.tick_params(labelsize = 14)
ax.set_ylim([0,1])

ax.plot([-0.5,1.5],[0.5,0.5],color="black",linestyle="dashdot")
ax.grid()

plt.show()

