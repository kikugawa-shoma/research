# coding: utf-8
import matplotlib.pyplot as plt
import numpy as np
import statistics

f = open("svm_DL_res.txt")
data = list(map(float,f.read().split()))
s = statistics.stdev(data)
err = [0 for i in range(len(data))]
err.append(s)
data.append(sum(data)/len(data))
x = [i for i in range(len(data))]
c = ["deepskyblue" for i in range(len(data))]
c[-1] = "red"
ytick = [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]



fig = plt.figure(figsize=(20,7))
ax = fig.add_subplot(1,1,1)
print(ax)
tick_label = [i for i in range(1,52)]
tick_label.append("mean")
ax.bar(x,data,color=c,yerr=err,capsize=5,edgecolor="black",linewidth=2,tick_label=tick_label)
ax.set_xlabel("Subjects",fontsize=25)
ax.set_ylabel("Decoding accuracy",fontsize=25)
ax.set_yticks(ytick)
ax.set_yticklabels(ytick, fontsize = 20)

ax.grid()

ax.plot([-1,52],[0.5,0.5],color="black",linestyle="dashdot")

plt.show()