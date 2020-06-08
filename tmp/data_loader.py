import scipy.io
import matplotlib.pyplot as plt
from sklearn.svm import SVC

filepath=r"C:\Users\Owner\ishii_lab\data\minibatch"
filename=r"\omp\minibatch_res_AtomN-1000_SparseDegree-100_MaxIter-100_dim-3530_sample_num-15504_alpha-1_batchsize-50.mat"
filepath += filename

data = scipy.io.loadmat(filepath)

