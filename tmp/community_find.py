"""
confusion matrix からaccuracy 60%以上を結合していると考えたグラフを計算し、そのグラフの
グラフクラスタリングを行うスクリプト
"""

import networkx as nx
import scipy.io
import numpy as np
import matplotlib.pyplot as plt
from networkx.algorithms import community 

G = scipy.io.loadmat(r"C:\Users\ktmks\Documents\my_matlab\make_figures\confusion_mat.mat")["confusion_mat"]
G = G > 60

plt.imshow(G)
#plt.show()

nodes = [i+1 for i in range(len(G))]  # 1-indexedに変えている(matlabでの図示と合わせるため)
edges = []
for i in range(1,len(G)+1):
    for j in range(1,i):
        if G[i-1,j-1]:  # 1-indexedに変えている(matlabでの図示と合わせるため)
            edges.append([i,j])

G = nx.Graph()
G.add_nodes_from(nodes)
G.add_edges_from(edges)

"""
被験者のグラフの描画
pos = nx.spring_layout(G)
nx.draw_networkx(G, pos, with_labels=True)
plt.axis("off")
plt.show()
"""


C = community.greedy_modularity_communities(G,2)  #グラフクラスタリング
C = list(C)
C = list(map(sorted,C))
print(list(map(sorted,C)))

label_n = len(C)
label = [0]*len(nodes)
for i in range(label_n):
    subject_N_in_label = len(C[i])
    for j in range(subject_N_in_label):
        label[C[i][j]-1] = i

"""
グラフクラスタリングによるラベルを保存
f = open("C:\\Users\\ktmks\\Documents\\research\\tmp\\results\\confusion_mat_classified_label.txt","w")
for i in range(len(nodes)):
    print(label[i],file=f,end=" ")
f.close()
"""