import PageRank as PR
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import matplotlib.pyplot as plt

pagerank = PR.PageRanks()

with open(r"results\confusion_mat_classified_label.txt") as f:
    all_label = np.array(list(map(int,f.readline().split())))

all_label = np.delete(all_label,14,0)
pr = np.delete(pagerank.pr,14,0)

clf = LinearDiscriminantAnalysis()
X = clf.fit_transform(pr,all_label)

color = ["red","blue","green"]
c = np.array([color[l] for l in all_label])
plt.scatter(list(X[:,0]),[0 for i in range(len(c))],c=c)
plt.show()

sig_ps_ind1 = np.load(r"results\e_num5\sig_ps.npy")
pr = pr[:,sig_ps_ind1]
clf = LinearDiscriminantAnalysis()
X = clf.fit_transform(pr,all_label)
plt.scatter(list(X[:,0]),[0 for i in range(len(c))],c=c)
plt.show()

acc = 0
predicted_label = []
for target in range(50):
    ttarget = target
    if ttarget >= 14:
        ttarget+=1
    sig_ps_ind = pagerank.ttest_significant_ind(target=ttarget,alpha=0.05,sampling="over",sample_diff=10)
    ind = np.array([i!=target for i in range(50)])
    pr = np.delete(pagerank.pr,14,0)[ind,:][:,sig_ps_ind]
    clf = LinearDiscriminantAnalysis()
    X = clf.fit_transform(pr,all_label[ind])
    x_target = clf.transform(np.delete(pagerank.pr,14,0)[target,sig_ps_ind].reshape(1,-1))
    pred = clf.predict(np.delete(pagerank.pr,14,0)[target,sig_ps_ind].reshape(1,-1))
    plt.scatter(list(X[:,0]),[0 for i in range(len(c)-1)],c=c[ind])
    if all_label[target] == 1:
        plt.scatter(x_target,0,c="aqua")
    else:
        plt.scatter(x_target,0,c="magenta")
    plt.show()
    if pred[0] == all_label[target]:
        acc += 1
    predicted_label.append(pred[0])
print(acc/len(all_label))
print("predited_label")
print(predicted_label)
print("label")
print(list(all_label))





