
import scipy.io
import numpy as np
import sklearn.metrics

subj_list = scipy.io.loadmat("C:\\Users\\ktmks\\Documents\\my_matlab\\use_subj.mat")["list"][0][:]
data_root = "C:\\Users\\ktmks\\Documents\\my_sources\\20\\"

T = []
f = open(r"C:\Users\ktmks\Documents\research\tmp\results\feature_values.txt",mode="r")
tmp = f.read().split("\n")
for i in range(len(subj_list)):
    T.append(list(map(float,tmp[i].split())))
f.close()

f = open("C:\\Users\\ktmks\\Documents\\research\\tmp\\results\\classified_label.txt")
L = list(map(int,f.readline().split()))
L_unique = list(set(L))

silhoutte_coeffs = sklearn.metrics.silhouette_samples(T,L)