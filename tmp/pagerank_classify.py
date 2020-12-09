from sklearn.svm import SVC
import numpy as np
import PageRank as PR
from collections import defaultdict
import copy
import community_find as cf
import matplotlib.pyplot as plt
import itertools as it

class PagerankDecoder(SVC):
    def __init__(self,C,gamma,kernel="rbf",class_weight=None):
        super().__init__(C=C,gamma=gamma,kernel=kernel,class_weight=class_weight)
    def fit(self,X,Y):
        X = np.array(X)
        Y = np.array(Y)
        self.N = len(Y)
        train_ind = [True if c != None else False for c in Y]
        train_x = X[train_ind,:]
        train_y = np.array(Y[train_ind],dtype=int)
        super().fit(train_x,train_y)
    def predict(self,x):
        return super().predict(x)


if __name__ == "__main__":

    gamma = ["scale","auto"]
    C = [1.0]
    class_weight = [None,"balanced"]
    sampling = ["over","under",None]
    sample_diff = [5*i for i in range(6)]
    params = []
    for ia in gamma:
        for ib in C:
            for ic in class_weight:
                for idd in sampling:
                    for ie in sample_diff:
                        params.append([ia,ib,ic,idd,ie])


    accuracy = []
    for i in range(len(params)):
        if i%10 == 0:
            print("{}/{}".format(i,len(params)))
        acc = 0
        predicted_label = []
        for target in range(51):
            # label = cf.ConfusionMatrix().community_detection_without(target)

            # This is the comment option
            with open(r"results\confusion_mat_classified_label.txt") as f:
                all_label = list(map(int,f.readline().split()))
            label = copy.copy(all_label)
            label[target] = None

            pagerank = PR.PageRanks()

            sig_ps_ind = pagerank.ttest_significant_ind(target = target,alpha=0.05,sampling=params[i][3],sample_diff=params[i][4])

            model = PagerankDecoder(C=params[i][1],gamma=params[i][0],class_weight=params[i][2])
            
            model.fit(np.delete(pagerank.pr[:,sig_ps_ind],14,0),np.delete(label,14,0))
            pred = model.predict(pagerank.pr[target,sig_ps_ind].reshape(1,-1))
            predicted_label.append(pred[0])
            if all_label[target] == pred[0]:
                acc += 1
        accuracy.append(acc/51)


