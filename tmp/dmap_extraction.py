import scipy.io
import h5py
import numpy as np
from tools.mapalign.embed import DiffusionMapEmbedding
import matplotlib.pyplot as plt

subj_list = scipy.io.loadmat("C:\\Users\\ktmks\\Documents\\my_matlab\\use_subj.mat")["list"][0][:]
des = []

for i in range(5):

    filepath = "C:\\Users\\ktmks\\Documents\\my_sources\\20\\{}\\rest\\regressed_out\\regressed_roi_brodmann_all_e_num5.mat"
    X = np.array(h5py.File(filepath.format(str(subj_list[i]).rjust(3,"0")),"r")["X"]).T
    dmapmodel = DiffusionMapEmbedding(alpha=0.5,
                                    diffusion_time = 10,
                                    affinity="markov",
                                    n_components=5)
    de = dmapmodel.fit_transform(X)
    ed = (de - de[0, :])
    ed = np.sqrt(np.sum(ed * ed , axis=1))
    ed = ed/max(ed)
    des.append(ed)