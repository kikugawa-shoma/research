import h5py
import networkx as nx
import numpy as np
import scipy.io
import matplotlib.pyplot as plt
import copy


class PR_Value():
    """
    Parameters
    ----------
    subj : 被験者の番号

    Attributes
    ----------
    p : numpy array, [nodes_num,nodes_num]
    r : numpy array, [nodes_num,nodes_num]
    N : int
        the number of rois or nodes
    """

    def __init__(self,subj,e_num):
        f = h5py.File(r"C:\Users\ktmks\Documents\my_sources\20" + r"\{:0>3}".format(subj) + r"\rest\regressed_out\regressed_inter_sl_con_roi_brodmann_all_e_num"+str(e_num)+".mat","r")
        self.p = np.array(f["Ps"][:][:])
        self.r = np.array(f["Rs"][:][:])
        self.N = len(self.p)
        self.substitute_nan()
    
    def substitute_nan(self):
        if sum(sum(np.isnan(self.p)))>0:
            print("There is some NaN voxels!")

        ps = []
        rs = []
        for i in range(self.N):
            for j in range(self.N):
                if not np.isnan(self.p[i][j]):
                    ps.append(self.p[i][j])
                    rs.append(self.r[i][j])
        mp = np.mean(ps)
        mr = np.mean(rs)
        stdp = np.std(ps)
        stdr = np.std(rs)
        for i in range(self.N):
            for j in range(self.N):
                if np.isnan(self.p[i][j]):
                    self.p[i][j] = np.random.normal(mp,stdp)
                    self.r[i][j] = np.random.normal(mr,stdr)

    #P値の行列から閾値で0/1の隣接行列を作る
    def make_adj_mat(self):
        """
        Parameters
        ----------

        Returns
        -------
        """
        self.adj_mat = self.p < (0.05/(self.N*(self.N-1)//2))
        for i in range(self.N):
            self.adj_mat[i][i] = False

    def create_network(self,directed=False,weighted=False):
        """
        Parameters
        ----------

        Returns
        -------
        Connectivity(G) : Connectivity object
        """
        self.make_adj_mat()
        nodes = [i for i in range(1,self.N+1)]
        edges = []
        for i in range(1,self.N):
            for j in range(1,self.N):
                if self.adj_mat[i-1][j-1]:
                    edges.append([i,j])
        G = nx.Graph()
        G.add_nodes_from(nodes)
        G.add_edges_from(edges)
        return Connectivity(G)

    def create_weighted_network(self,directed=False,weighted=False):
        """
        Parameters
        ----------

        Returns
        -------
        Connectivity(G) : Connectivity object
        """
        self.make_adj_mat()

        nodes = [i for i in range(1,self.N+1)]
        edges = []

        G = nx.Graph()
        G.add_nodes_from(nodes)
        G.add_edges_from(edges)
        for i in range(1,self.N):
            for j in range(1,self.N):
                if i>j:
                    G.add_edge(i,j,weight=abs(self.r[i][j]))
        return Connectivity(G)
    

class Connectivity():
    def __init__(self,nxgraph):
        self.G = nxgraph
    
    def pagerank(self):
        pr = [0]*len(self.G.nodes)
        dict_pagerank = nx.pagerank(self.G)
        for i in range(len(self.G.nodes)):
            pr[i] = dict_pagerank[i+1]
        return pr

if __name__ == "__main__":
    subj_list = scipy.io.loadmat("C:\\Users\\ktmks\\Documents\\my_matlab\\use_subj.mat")["list"][0][:]
    w_pageranks = [] # 相関係数の重みありで計算したpagerank
    pageranks = [] # 重みなしで計算しpagerank

    for cnt,subj in enumerate(subj_list):
        print(subj)
        pr_value = PR_Value(subj,e_num=5)
        w_conn = pr_value.create_weighted_network()
        w_pageranks.append(w_conn.pagerank())

        conn = pr_value.create_network()
        pageranks.append(conn.pagerank())
    pageranks = np.array(pageranks)
    w_pageranks = np.array(w_pageranks)
    np.savez("results\\pageranks",pageranks=pageranks,w_pageranks=w_pageranks)

