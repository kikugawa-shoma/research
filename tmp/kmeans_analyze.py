import scipy.io
from svm_raw_nearests_decoding import PageRanks as PR
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import itertools

label = scipy.io.loadmat(r"C:\Users\ktmks\Documents\my_matlab\make_figures\kmeans.mat")["label1"]
pageranks = PR()


def t_test(labels,label1,label2,pagerank,bin=10):
    pr_l1 = []
    pr_l2 = []
    for i in range(len(pageranks.pr)):
        if labels[i] == label1:
            pr_l1.append(pageranks.pr[i])
        if labels[i] == label2:
            pr_l2.append(pageranks.pr[i])
    ts,ps = stats.ttest_ind(pr_l1,pr_l2)

    plt.hist(ps,bins=bin)
    plt.plot([0,1],[len(ps)/bin for _ in range(2)],linestyle="dashed",color="black")
    plt.show()
    return ts,ps


for c in itertools.combinations([1,2,3],2):
    t_test(label,c[0],c[1],pageranks.pr)


