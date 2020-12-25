import CorrPvalue as CP
from tools.mapalign.embed import DiffusionMapEmbedding
import matplotlib.pyplot as plt
import scipy.io
import copy
import numpy as np
import networkx as nx

subj_list = scipy.io.loadmat("C:\\Users\\ktmks\\Documents\\my_matlab\\use_subj.mat")["list"][0][:]


class Dmap_r(CP.P_Value):
    def __init__(self,subj):
        super().__init__(subj)

    def dmap_prepro(self,remain_ratio=10):
        self.tmp = self.r
        for i in range(self.N):
            sorted_ind = np.argsort(self.p[i])[::-1]
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

des = []
for subj in subj_list:
#for subj in [subj_list[48]]:
    if subj == 32:
        continue # because subject 32 data contain NaN
    dmap_r = Dmap_r(subj)
    dmap_r.dmap_prepro()
    dmapmodel = DiffusionMapEmbedding(alpha=0.5,
                                 diffusion_time = 10,
                                 affinity="precomputed",
                                 n_components=5)
    de = dmapmodel.fit_transform(dmap_r.tmp)
    des.append(de)
    print(subj)
np.save(r"results\dmap_feature.npy",des)
    #dmapmodel.fit_transform(copy.deepcopy(pr.r))

    


