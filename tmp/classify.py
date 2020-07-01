"""
コネクティビティを表すグラフから得られた特徴量データをロードして
それに基づいてk-meansクラスタリングにより被験者をグループに分けたラベルを生成
"""
import scipy.io
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

subj_list = scipy.io.loadmat("C:\\Users\\ktmks\\Documents\\my_matlab\\use_subj.mat")["list"][0][:]
data_root = "C:\\Users\\ktmks\\Documents\\my_sources\\20\\"

T = [[0]*2 for _ in range(len(subj_list))]

for i,subj in enumerate(subj_list):
    id = "{:0>3}".format(subj)
    subj_data_path = data_root + id + "\\rest\\inter_feature.mat"
    T[i] = scipy.io.loadmat(subj_data_path)["x"][0][:]

"""
T = np.array(T)
label = KMeans(n_clusters=2).fit_predict(T)
plt.scatter(T[:,0],T[:,1],c=label)
plt.show()
"""






