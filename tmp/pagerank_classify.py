from sklearn.svm import SVC
import numpy as np
from svm_raw_nearests_decoding import PageRanks as PR
from collections import defaultdict
import copy

class PagerankDecoder(SVC):
    def __init__(self,C,gamma,kernel="rbf"):
        super().__init__(kernel=kernel,class_weight="balanced",C=C,gamma=gamma)
    def fit_predict(self,X,Y,test_ind):
        X = np.array(X)
        Y = np.array(Y)
        self.N = len(Y)
        train_ind = [True]*self.N
        train_ind[test_ind] = False
        self.train_x = X[train_ind,:]
        self.train_y = Y[train_ind]
        self.test_x = X[test_ind,:]
        self.test_y = Y[test_ind]
        super().fit(self.train_x,self.train_y)
        return int(super().predict(self.test_x.reshape(1,-1)))


if __name__ == "__main__":
    with open(r"results\confusion_mat_classified_label.txt") as f:
        label = list(map(int,f.readline().split()))
    
    ps = np.load(r"results\weighted_pagerank_p-value.npy")
    sig_ps = ps<0.05
    predicted_label = []
    for i in range(len(label)):
        model = PagerankDecoder(C=1.0,gamma="scale")
        predicted_label.append(model.fit_predict(PR().pr[:,sig_ps],label,i))
    predicted_label = np.array(predicted_label)
    label = np.array(label)
    print(sum(label == predicted_label)/51)
