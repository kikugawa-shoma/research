import h5py
import networkx as nx
import numpy as np
import scipy.io


class P_Value():
    """
    Parameters
    ----------
    subj : 被験者の番号

    Attributes
    ----------
    p : numpy array, [nodes_num,nodes_num]
    N : int
        the number of rois or nodes
    """

    def __init__(self,subj):
        f = h5py.File(r"C:\Users\ktmks\Documents\my_sources\20" + r"\{:0>3}".format(subj) + r"\rest\regressed_out\regressed_inter_sl_con.mat","r")
        self.p = np.array(f["Ps"][:][:])
        self.N = len(self.p)
    
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
    
class R_value():
    def __init__(self):
        f = h5py.File(r"C:\Users\ktmks\Documents\my_sources\20" + r"\{:0>3}".format(subj) + r"\rest\regressed_out\regressed_inter_sl_con.mat","r")
        self.r = f["Rs"].value
    
    def create_network(self):
        pass

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
    pageranks = []
    for cnt,subj in enumerate(subj_list):
        p_value = P_Value(subj)
        conn = p_value.create_network()
        pageranks.append(conn.pagerank())
    pageranks = np.array(pageranks)

    #保存
    savepath = r"C:\Users\ktmks\Documents\research\tmp\results\feature_values.txt"
    with open(savepath,mode="w") as f:
        for i in range(len(pageranks)):
            for j in range(len(pageranks[0])):
                f.write(str(pageranks[i][j])+" ")
            f.write("\n")
