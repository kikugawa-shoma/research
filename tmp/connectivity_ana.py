import CorrPvalue as CP
import scipy.io
import matplotlib.pyplot as plt
import svm_raw_nearests_decoding as PR
import numpy as np
from scipy import stats
from collections import defaultdict

#データの前準備
class_label_path = r"C:\Users\ktmks\Documents\research\tmp\results\confusion_mat_classified_label.txt"
with open(class_label_path,mode="r") as f:
    labels = list(map(int,f.read().split()))

subj_list = scipy.io.loadmat("C:\\Users\\ktmks\\Documents\\my_matlab\\use_subj.mat")["list"][0][:]

pagerank = PR.PageRanks()

#conversion matrixによるクラスタリングでのグループ間の各roiのpagerankの平均に関するt検定
c_pagerank = [[] for _ in range(len(set(labels)))]
for i in range(pagerank.N):
    L = labels[i]
    c_pagerank[L].append(pagerank.pr[i])

ts,ps = stats.ttest_ind(c_pagerank[0],c_pagerank[1])

plt.hist(ps,bins=8)
plt.show()


#ランダムシャッフルによるクラスタリングでのグループ間の各roiのpagerankの平均に関するt検定
c_pagerank = [[] for _ in range(len(set(labels)))]
random_labels = np.random.randint(0,2,[51])
for i in range(pagerank.N):
    L = random_labels[i]
    c_pagerank[L].append(pagerank.pr[i])

rand_label_ts,rand_label_ps = stats.ttest_ind(c_pagerank[0],c_pagerank[1])

plt.hist(rand_label_ps,bins=15)
plt.show()




"""
#conversion matrixによるクラスタリングでのグループ間の各roiのコネクティビティの平均に関するt検定
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
"""
