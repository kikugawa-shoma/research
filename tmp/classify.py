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

T = []
f = open(r"C:\Users\ktmks\Documents\research\tmp\results\feature_values.txt",mode="r")
tmp = f.read().split("\n")
for i in range(len(subj_list)):
    T.append(list(map(float,tmp[i].split())))
f.close()



T = np.array(T)
label = KMeans(n_clusters=2).fit_predict(T)

save_path = "C:\\Users\\ktmks\\Documents\\research\\tmp\\results\\classified_label.txt"
with open(save_path,"w") as f:
    for i in range(len(subj_list)):
        print(label[i],file=f,end=" ")

"""
plt.scatter(T[:,0],T[:,1],c=label)
plt.show()
"""






