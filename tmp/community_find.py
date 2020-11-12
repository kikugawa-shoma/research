"""
confusion matrix からaccuracy 60%以上を結合していると考えたグラフを計算し、そのグラフの
グラフクラスタリングを行うスクリプト
"""

import networkx as nx
import scipy.io
import numpy as np
import matplotlib.pyplot as plt
from networkx.algorithms import community 
import copy

class confusion_mat():
    """
    one training one testでのaccuracyを並べたconfusion matrixから
    隣接行列をthleasholdで0/1に変換することにより計算するためのクラス

    Parameters
    -----------
    thleashold : 

    Attributes
    -----------
    accuracy_mat : 
    adj_mat      : 
    N            : 

    """
    def __init__(self,thleashold = 60):
        self.accuracy_mat = scipy.io.loadmat(r"C:\Users\ktmks\Documents\my_matlab\make_figures\confusion_mat.mat")["confusion_mat"]
        self.adj_mat = self.accuracy_mat > thleashold
        self.N = len(self.adj_mat)
    
    def make_graph_without(self,subj):
        """
        subjを孤立ノードとしたグラフを構築しattributeのself.Gへ追加する関数

        Parameters
        ----------
        subj : 取り除くべきsubjectのindex(target subject)


        """
        subtracted_G = copy.deepcopy(self.adj_mat)
        for i in range(self.N):
            subtracted_G[subj][i] = 0
            subtracted_G[i][subj] = 0
        self.nodes = [i for i in range(self.N)] 
        self.edges = []
        for i in range(self.N):
            for j in range(i):
                if subtracted_G[i,j]:  
                    self.edges.append([i,j])

        self.G = nx.Graph()
        self.G.add_nodes_from(self.nodes)
        self.G.add_edges_from(self.edges)

        """
        描画(デバッグ用)
        pos = nx.spring_layout(self.G)
        nx.draw_networkx(self.G, pos, with_labels=True)
        plt.axis("off")
        plt.show()
        """

    def community_detection_without(self,subj):
        """
        self.Gからcommunity detectionによりグラフクラスタリングを実行する関数

        Parameters
        ----------

        Returns
        ----------
        label : list, [1,self.N]
        """
        self.make_graph_without(subj)
        C = community.greedy_modularity_communities(self.G,2)  #グラフクラスタリング
        C = list(C)
        C = list(map(sorted,C))

        label_n = len(C)
        label = [0]*len(self.nodes)
        for i in range(label_n):
            subject_N_in_label = len(C[i])
            for j in range(subject_N_in_label):
                label[C[i][j]] = i
        label[subj] = None
        return label

if __name__ == "__main__":
    for i in range(51):
        A = confusion_mat()
        print(i+1,A.community_detection_without(i))

