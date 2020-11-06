import CorrPvalue as CP
from tools.mapalign.embed import DiffusionMapEmbedding
import matplotlib.pyplot as plt
import scipy.io
import copy
import numpy as np

subj_list = scipy.io.loadmat("C:\\Users\\ktmks\\Documents\\my_matlab\\use_subj.mat")["list"][0][:]


class Dmap_r(CP.P_Value):
    def __init__(self,subj):
        super().__init__(subj)

    def dmap_prepro(self,remain_ratio=10):
        self.tmp = copy.deepcopy(self.r)
        plt.imshow(self.tmp[1:50,1:50])
        plt.show()
        for i in range(self.N):
            sort_ind = np.argsort(self.r[i])
            sort_ind = sort_ind[::-1]
            for j in range(self.N):
                if j > self.N*remain_ratio/100:
                    self.tmp[i][sort_ind[j]] = 0
                elif self.tmp[i][sort_ind[j]] < 0:
                    self.tmp[i][sort_ind[j]] = 0
        plt.imshow(self.tmp)
        plt.show()
                    

des = []
for subj in subj_list:
    if subj == 32:
        continue # because subject 32 data contain NaN
    dmap_r = Dmap_r(subj)
    dmapmodel = DiffusionMapEmbedding(alpha=0.5,
                                 diffusion_time = 10,
                                 affinity="nearest_neighbors",
                                 n_components=5)
    de = dmapmodel.fit_transform(dmap_r.r)
    des.append(de)
np.save(r"results\dmap_feature.npy",des)
    #dmapmodel.fit_transform(copy.deepcopy(pr.r))

    


