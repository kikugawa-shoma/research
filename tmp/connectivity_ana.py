import CorrPvalue as CP
import scipy.io
import matplotlib.pyplot as plt
import svm_raw_nearests_decoding as PR
import numpy as np
from scipy import stats
from collections import defaultdict

"""
1 : conversion matrixから得られたラベルでグループ分けされた被験者群間の各サーチライトに対してpagerankのt検定を行い、
    その結果をヒストグラムに表示
2 : 1の結果のヒストグラムに対して帰無仮説を一様分布とした適合度検定を行う
"""

#一様分布との適合度検定を行う関数
def chi_squared_test(xs,bin_n = 10,a = 0,b = 1):
    """
    一様分布を帰無仮説とした適合度検定を行う関数

    parameters
    --------------
    xs    : list[データの総数,1]
    bin_n : binの個数
    a     : データの下限値
    b     : データの上限値

    returns
    --------------
    p_value : 一様分布との適合度検定のp値
    """
    hist = [0]*bin_n
    N = len(xs)
    ab = b-a
    d = ab*(1/bin_n)
    for x in xs:
        for k in range(bin_n):
            if a+d*k <= x < a+d*(k+1):
                hist[k]+=1
    chi2_0 = 0
    ei = N/bin_n
    for k in range(bin_n):
        chi2_0 += ((hist[k]-ei)**2)/ei
    p_value = stats.chi2.sf(x = chi2_0,df = bin_n-1)
    return p_value

#データの前準備
labels = list(scipy.io.loadmat(r"C:\Users\ktmks\Documents\my_matlab\make_figures\kmeans.mat")["label1"][:,0])
for i in range(len(labels)):
    if labels[i] == 1:
        labels[i] = 0
    elif labels[i] == 2:
        labels[i] = 1
    elif labels[i] == 3:
        labels[i] = 1


subj_list = scipy.io.loadmat("C:\\Users\\ktmks\\Documents\\my_matlab\\use_subj.mat")["list"][0][:]

pagerank = PR.PageRanks(weighted=True)

#conversion matrixによるクラスタリングでのグループ間の各roiのpagerankの平均に関するt検定
c_pagerank = [[] for _ in range(len(set(labels)))]
for i in range(pagerank.N):
    L = labels[i]
    c_pagerank[L].append(pagerank.pr[i])

ts,ps = stats.ttest_ind(c_pagerank[0],c_pagerank[1])

bins = 10
plt.hist(ps,bins=bins)
plt.plot([0,1],[len(ps)/bins,len(ps)/bins],linestyle="dashed",color="black")
plt.show()



#χ二乗分布による適合度検定
p = chi_squared_test(ps,bin_n=10)
p_rand = chi_squared_test(rand_label_ps)
print("confusion_mat classified : {}\nrandom classified : {}".format(p,p_rand))


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
