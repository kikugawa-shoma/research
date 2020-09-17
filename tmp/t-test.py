import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from statistics import mean, stdev

x = []
root = "C:\\Users\\ktmks\\Documents\\research\\tmp\\results\\"
filenames = [
    "svm_raw",
    "svm_raw_confusion_mat_classified",
    "svm_raw_classified",
    ]

for i in range(len(filenames)):
    f = open(root+filenames[i]+".txt","r")
    x.append(list(map(float,f.readline().split())))
    f.close()

res = stats.ttest_rel(x[1],x[0])
print(res)

colorlist = ["r", "g", "b", "c", "m", "y", "k"]

fig = plt.figure()
ax = fig.add_subplot(111)
"""
for i in range(len(x[0])):
    ax.plot([0,1],[x[0][i],x[1][i]],color=colorlist[i%7])
    ax.plot([1,2],[x[1][i],x[2][i]],color=colorlist[i%7])
"""

bar_y = []
err = []

for i in range(len(filenames)):
    bar_y.append(mean(x[i]))
    err.append(stdev(x[i]))
ax.bar([0,1,2],bar_y,width=0.6,yerr=err,capsize=10)
ax.set_ylim([0.3,1])

plt.show()






