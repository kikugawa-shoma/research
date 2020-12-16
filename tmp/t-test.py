import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from statistics import mean, stdev

predicted_label = np.load("results\\pagerank_classified_label.npy")

x = []
root = "C:\\Users\\ktmks\\Documents\\research\\tmp\\results\\"
filenames = [
    "svm_raw",
    "svm_raw_pagerank_classified",
    ]

for i in range(len(filenames)):
    f = open(root+filenames[i]+".txt","r")
    x.append(list(map(float,f.readline().split())))
    f.close()

for i in range(len(x[0])):
    if predicted_label[i] == 1:
        x[1][i] = x[0][i]

res = stats.ttest_rel(x[1],x[0])
print(res)

colorlist = ["red","blue"]

fig = plt.figure()
ax = fig.add_subplot(111)
for i in range(len(x[0])):
    ax.plot([0,1],[x[0][i],x[1][i]],color=colorlist[predicted_label[i]])

bar_y = []
err = []

for i in range(len(filenames)):
    bar_y.append(mean(x[i]))
    err.append(stdev(x[i]))
ax.bar([0,1],bar_y,width=0.6,yerr=err,capsize=10)
ax.set_ylim([0.3,1])

plt.show()






