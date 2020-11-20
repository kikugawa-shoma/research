import  matplotlib.pyplot as plt

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

class ConfusionMatrix():
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
        # confusion matrixをロード
        self.accuracy_mat = scipy.io.loadmat(r"C:\Users\ktmks\Documents\my_matlab\make_figures\confusion_mat.mat")["confusion_mat"]

        # thleacholdで0/1の隣接行列を作成
        self.adj_mat = self.accuracy_mat > thleashold

        self.N = len(self.adj_mat)
    
    def make_graph_without(self,subj):
        """
        subjノードと孤立ノードを削除したグラフを構築しattributeのself.Gへ追加する関数

        Parameters
        ----------
        subj : 取り除くべきsubjectのindex(target subject)


        """
        # グラフの構築
        self.nodes = [i for i in range(self.N)] 
        self.edges = []
        for i in range(self.N):
            for j in range(i):
                if self.adj_mat[i,j]:  
                    self.edges.append([i,j])
        self.G = nx.Graph()
        self.G.add_nodes_from(self.nodes)
        self.G.add_edges_from(self.edges)

        # subjノードを削除
        self.G.remove_node(subj)

        # 孤立ノードを削除
        isolated = [k for k,v in self.G.degree() if v == 0]
        for iso_node in isolated:
            self.G.remove_node(iso_node)

        """
        描画(デバッグ用)
        pos = nx.spring_layout(self.G)
        nx.draw_networkx(self.G, pos, with_labels=True)
        plt.axis("off")
        plt.show()
        """

    def community_detection_without(self,subj):
        """
        subjノードと孤立ノードを取り除いたうえでself.Gからcommunity detectionにより
        グラフクラスタリングを実行する関数

        Parameters
        ----------
        subj : 孤立ノードとする被験者のindex

        Returns
        ----------
        label : list, [1,self.N]
        """

        self.make_graph_without(subj)

        # グラフクラスタリング
        C = community.greedy_modularity_communities(self.G)
        C = list(C)
        C = list(map(sorted,C))

        label = [None]*self.N
        for ind_c in range(len(C)):
            for node in C[ind_c]:
                label[node] = ind_c
        
        """%1
        描画のためにNoneを取り除く
        label[label.index(None)] = 3
        if None in label:
            label[label.index(None)] = 3
        """

        return label

if __name__ == "__main__":

    tmp = []
    for i in range(51):
        A = ConfusionMatrix()
        print(i,A.community_detection_without(i))
        tmp.append(A.community_detection_without(i))

    """%1
    plt.imshow(tmp)
    plt.show()
    """


