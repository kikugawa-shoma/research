import CorrPvalue as CP
import scipy.io
import matplotlib.pyplot as plt

class_label_path = r"C:\Users\ktmks\Documents\research\tmp\results\confusion_mat_classified_label.txt"
with open(class_label_path,mode="r") as f:
    labels = list(map(int,f.read().split()))

subj_list = scipy.io.loadmat("C:\\Users\\ktmks\\Documents\\my_matlab\\use_subj.mat")["list"][0][:]

p_vals = []
for i in range(len(subj_list)):
    p_vals.append(CP.P_Value(subj_list[0]))


