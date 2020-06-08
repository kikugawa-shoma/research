import scipy.io
import matplotlib.pyplot as plt
from sklearn.svm import SVC

par = [3,5,10,20,30,40,50]
data = [0]*len(par)

filepath=r"C:\Users\Owner\ishii_lab\data\minibatch"
for i in range(len(par)):
    filename=r"\omp\minibatch_res_AtomN-1000_SparseDegree-100_MaxIter-100_dim-3530_sample_num-15504_alpha-1_batchsize-" + str(par[i]) + r".mat"
    file = filepath + filename

    data[i] = scipy.io.loadmat(file)

    

