from sklearn.decomposition import DictionaryLearning
import scipy.io
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import dill

N_COMPONENTS               =  500
TRANSFORM_N_NONZERO_COEFS  =   10
VERBOSE                    =  True
MAX_ITER                   =   10

MatBrainImage=scipy.io.loadmat(r"C:\Users\ktmks\Documents\research\tmp_results\for_python_data\brain_f_data.mat")

label=MatBrainImage["label"]
Y=MatBrainImage["data"]

dic=DictionaryLearning(n_components              =              N_COMPONENTS,
                       transform_n_nonzero_coefs = TRANSFORM_N_NONZERO_COEFS,
                       verbose                   =                   VERBOSE,
                       max_iter                  =                  MAX_ITER 
                       )
dic.fit(Y)
D=dic.components_
X=dic.transform(Y)
Y_=np.dot(X,D)

filepath = r"C:\Users\ktmks\Documents\research\Python\Brain_DL"+"\\"
filename = "res_"+"AtomN-"   + str(N_COMPONENTS)\
          +"_SparseDegree-"  + str(TRANSFORM_N_NONZERO_COEFS)\
          +"_MaxIter-"       + str(MAX_ITER)
save_filename=filepath+filename+".pkl"

dill.dump_session(save_filename)
scipy.io.savemat(filename+".mat",{"D":D,"X":X,"Y_":Y_,"label":label})