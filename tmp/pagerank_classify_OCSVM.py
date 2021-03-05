from sklearn.svm import OneClassSVM
import numpy as np
import PageRank as PR
from collections import defaultdict
import copy
import community_find as cf
import matplotlib.pyplot as plt
from matplotlib import pyplot as plt




pagerank_kind = "p" # "p" or "w" or "wt"
with open(r"results\confusion_mat_classified_label.txt") as f:
    all_label = list(map(int,f.readline().split()))

feature_value = PR.PageRanks(weighted=pagerank_kind)

positive = sum([l == 0 for l in all_label])
negative = sum([l != 0 for l in all_label])

TPRs = []
TNRs = []
sig_ps_ind = np.load("results//e_num5//sig_ps.npy")
for i in range(1,11):
    model = OneClassSVM(nu=i/10)
    pred = model.fit_predict(feature_value.pr[:,sig_ps_ind])
    pred = [0 if p == 1 else 1 for p in pred]
    print(pred)
    TPR = 0
    TNR = 0
    for j in range(len(pred)):
        if all_label[j] == 0 and pred[j] == 0:
            TPR += 1
        if all_label[j] == 1 and pred[j] == 1:
            TNR += 1
    TPR /= positive
    TNR /= negative
    TPRs.append(TPR)
    TNRs.append(TNR)

plt.plot(TNRs,TPRs)
plt.plot([0,1],[1,0],c="red")
plt.show()




