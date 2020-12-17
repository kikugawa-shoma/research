from sklearn.svm import SVC
import numpy as np
import PageRank as PR
from collections import defaultdict
import copy
import community_find as cf
import matplotlib.pyplot as plt
import scipy

if __name__ == "__main__":
    accuracy = 0
    sig_img = []
    predicted_label = []
    all_label = list(scipy.io.loadmat(r"C:\Users\ktmks\Documents\my_matlab\make_figures\kmeans.mat")["label1"][:,0])
    for i in range(len(all_label)):
        if all_label[i] == 1:
            all_label[i] = 0
        elif all_label[i] == 2:
            all_label[i] = 1
        elif all_label[i] == 3:
            all_label[i] = 1
    """
    with open("results\\confusion_mat_classified_label.txt") as f:
        all_label = list(map(int,f.readline().split()))
    """
    all_label = np.array(all_label)
    for target in range(51):

        pagerank = PR.PageRanks(weighted="w")

        train_ind = np.array([True if i!=target else False for i in range(51)])

        sig_ps_ind = pagerank.ttest_significant_ind(target=target)

        d = {1:sum(all_label==0),0:sum(all_label==1)}

        model = SVC(C=1,gamma="scale",kernel="poly",class_weight="balanced",degree=7)
        model.fit(pagerank.pr[train_ind,:][:,sig_ps_ind],all_label[train_ind])

        for i in range(51):
            print(model.predict(pagerank.pr[i,][sig_ps_ind].reshape(1,-1))[0],end=" ")
        print("")

        pred = model.predict(pagerank.pr[target,:][sig_ps_ind].reshape(1,-1))
        predicted_label.append(pred[0])
        if all_label[target] == pred[0]:
            accuracy += 1
    accuracy = accuracy/51
    print(accuracy)
    print(np.array(predicted_label))
    print(all_label)
