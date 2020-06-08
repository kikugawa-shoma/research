import statistics
import matplotlib.pyplot as plt

f = open("svm_DL_param_search_res.txt","r")
data = f.read().split("\n")
data = [list(map(float, d.split()[2:])) for d in data]
del data[-1]
err = []
for d in data:
    d.append(statistics.mean(d[1:52]))
    err.append(statistics.stdev(d[1:52]))
x_label = [str(d[0]) for d in data]
x = [i for i in range(len(x_label))]
y = [d[-1] for d in data]

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.bar(x,y,yerr=err,capsize=5,tick_label=x_label)
ax.plot([-1,8],[0.5,0.5],color="black",linestyle="dashdot")
plt.show()





