import CorrPvalue as CP
import scipy.io
import matplotlib.pyplot as plt
import svm_raw_nearests_decoding as PR
import numpy as np
from scipy import stats
from collections import defaultdict


class_label_path = r"C:\Users\ktmks\Documents\research\tmp\results\confusion_mat_classified_label.txt"
with open(class_label_path,mode="r") as f:
    labels = list(map(int,f.read().split()))

subj_list = scipy.io.loadmat("C:\\Users\\ktmks\\Documents\\my_matlab\\use_subj.mat")["list"][0][:]

pagerank = PR.PageRanks()

c_pagerank = [[] for _ in range(len(set(labels)))]
for i in range(pagerank.N):
    L = labels[i]
    c_pagerank[L].append(pagerank.pr[i])

ts,ps = stats.ttest_ind(c_pagerank[0],c_pagerank[1])

plt.hist(ps,bins=10)
plt.show()




rand = np.random.rand(np.shape(pagerank.pr)[0],np.shape(pagerank.pr)[1])
c_rand = [[] for _ in range(len(set(labels)))]
for i in range(pagerank.N):
    L = labels[i]
    c_rand[L].append(rand[i])

rand_ts,rand_ps = stats.ttest_ind(c_rand[0],c_rand[1])

plt.hist(rand_ps,bins=10)
plt.show()


c_con = [[] for _ in range(len(set(labels)))]
for i in range(len(subj_list)):
    tmp = CP.P_Value(subj_list[i]).p
    tmp = np.extract(1-np.eye(len(tmp)),tmp)
    if sum(np.isnan(tmp)) != 0:
        print(i,subj_list[i])
    else:
        c_con[labels[i]].append(tmp)
    
con_ts,con_ps = stats.ttest_ind(c_con[0],c_con[1])
plt.hist(con_ps,bins=500)
plt.show()









    







