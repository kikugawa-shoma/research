import scipy.io
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

def feature_value_ttest(dmap_ind,alpha=0.05):
    #データの前準備
    class_label_path = r"C:\Users\ktmks\Documents\research\tmp\results\confusion_mat_classified_label.txt"
    with open(class_label_path,mode="r") as f:
        labels = list(map(int,f.read().split()))
    """
    labels = list(scipy.io.loadmat("C:\\Users\\ktmks\\Documents\\my_matlab\\make_figures\\kmeans.mat")["label1"][:,0])
    for i in range(len(labels)):
        if labels[i] ==  1:
            labels[i] = 0
        elif labels[i] == 2:
            labels[i] = 1
        elif labels[i] == 3:
            labels[i] = 1
    """

    subj_list = scipy.io.loadmat("C:\\Users\\ktmks\\Documents\\my_matlab\\use_subj.mat")["list"][0][:]

    tmp = np.load("results/dmap_feature.npy")
    feature_value = []
    for i in range(len(tmp)):
        feature_value.append(tmp[i][:,dmap_ind])

    #conversion matrixによるクラスタリングでのグループ間の各roiのpagerankの平均に関するt検定
    c_feature_value = [[] for _ in range(len(set(labels)))]
    for i in range(len(feature_value)):
        L = labels[i]
        c_feature_value[L].append(feature_value[i])

    _,ps = stats.ttest_ind(c_feature_value[0],c_feature_value[1])

    bins = 20
    plt.hist(ps,bins=bins)
    plt.plot([0,1],[len(ps)/bins,len(ps)/bins],linestyle="dashed",color="black")
    plt.show()
    print(sum(ps<0.05)/len(ps))
    return ps<alpha

if __name__=="__main__":
    sig_ps_ind = feature_value_ttest(1)
