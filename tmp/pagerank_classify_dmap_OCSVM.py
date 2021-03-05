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

feature_value = PR.PageRanks(weighted=pagerank_kind,feature="dmap")


positive = sum([l == 0 for l in all_label])
negative = sum([l != 0 for l in all_label])

TPRs = []
TNRs = []
for i in range(1,11):
    TPR = 0
    TNR = 0
    for target in range(51):
        nu = i/10
        model = OneClassSVM(nu=nu)
        sig_ps_ind = feature_value.ttest_significant_ind(target,alpha=0.03,sampling=None,sample_diff=1)
        all_pred = model.fit_predict(feature_value.pr[:,sig_ps_ind])
        if all_pred[target] == -1:
            pred = 1
        else:
            pred = 0
        
        target_label = all_label[target]

        if target_label == 0 and pred == 0:
            TPR += 1
        if target_label == 1 and pred == 1:
            TNR += 1
    TPRs.append(TPR/positive)
    TNRs.append(TNR/negative)

plt.plot(TNRs,TPRs)
plt.plot([0,1],[1,0],c="red")
plt.show()




