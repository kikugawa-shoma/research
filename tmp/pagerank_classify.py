from sklearn.svm import SVC
import numpy as np
import PageRank as PR
from collections import defaultdict
import copy
import community_find as cf

class PagerankDecoder(SVC):
    def __init__(self,kernel="rbf"):
        super().__init__(kernel=kernel)
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
    accuracy = 0
    for target in range(51):
        label = cf.ConfusionMatrix().community_detection_without(target)
        print(label)

        pagerank = PR.PageRanks()
        
        sig_ps_ind = PR.PageRanks().ttest_significant_ind(target = target)

        model = PagerankDecoder()
        model.fit(pagerank.pr[:,sig_ps_ind],label)
        pred = model.predict(pagerank.pr[target,sig_ps_ind].reshape(1,-1))
        print(target,label[target],pred)
        if label[target] == pred[0]:
            accuracy += 1
    accuracy = accuracy/51
    print(accuracy)

    """
    predicted_label = []
    for i in range(len(label)):
        model = PagerankDecoder()
        predicted_label.append(model.fit_predict(PR().pr[:,sig_ps],label,i))
    predicted_label = np.array(predicted_label)
    label = np.array(label)
    print(sum(label == predicted_label)/51)
    """

    
    


