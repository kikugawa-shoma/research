import CorrPvalue as CP
from tools.mapalign.embed import DiffusionMapEmbedding
import matplotlib.pyplot as plt
import scipy.io
import copy
import numpy as np
import networkx as nx
import connectivity_ana as ca

subj_list = scipy.io.loadmat("C:\\Users\\ktmks\\Documents\\my_matlab\\use_subj.mat")["list"][0][:]


class Dmap_r(CP.P_Value):
    def __init__(self,subj):
        super().__init__(subj)

    def dmap_prepro(self,remain_ratio=50):
        #self.tmp = copy.deepcopy(self.r)
        self.tmp = np.zeros(self.r.shape)
        for i in range(self.N):
            sorted_ind = np.argsort(self.p[i])
            prop = int(self.N*(remain_ratio/100))
            for cnt in range(prop):
                if sorted_ind[cnt] == i:
                    continue
                self.tmp[i][sorted_ind[cnt]] = 1
        
    
    def make_G(self):
        self.G = nx.Graph()
        for i in range(self.N):
            self.G.add_node(i)
        for i in range(self.N):
            for j in range(self.N):
                if self.tmp[i][j]:
                    self.G.add_edge(i,j)

def diffusion_map_embed(alpha,diffusion_time):
    des = []
    for subj in subj_list:
        dmap_r = Dmap_r(subj)
        dmap_r.dmap_prepro()
        dmapmodel = DiffusionMapEmbedding(
                                          alpha          = alpha,
                                          diffusion_time = diffusion_time,
                                          affinity       = "precomputed",
                                          n_components   = 5
                                          )
        de = dmapmodel.fit_transform(dmap_r.tmp)
        des.append(de)
        print(subj)
    np.save(r"results\dmap_feature.npy",des)
        #dmapmodel.fit_transform(copy.deepcopy(pr.r))

if __name__=="__main__":
    diffusion_map_embed(0.5,0)
        
