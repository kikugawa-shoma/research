from sklearn.svm import SVC
import numpy as np
import PageRank as PR
from collections import defaultdict
import copy
import community_find as cf
import matplotlib.pyplot as plt

class PagerankDecoder(SVC):
    def __init__(self,kernel="rbf",class_weight=None):
        super().__init__(kernel=kernel,class_weight=class_weight)
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
    sig_img = []
    for target in range(51):
        # label = cf.ConfusionMatrix().community_detection_without(target)

        # This is the comment option
        with open(r"results\confusion_mat_classified_label.txt") as f:
            all_label = list(map(int,f.readline().split()))
        label = copy.copy(all_label)
        label[target] = None

        pagerank = PR.PageRanks()
        
        sig_ps_ind = PR.PageRanks().ttest_significant_ind(target = target)
        sig_ps_ind1 = np.load(r"results\e_num5\sig_ps.npy")
        diff = 0
        for i in range(len(sig_ps_ind)):
            if sig_ps_ind[i]:
                if not sig_ps_ind1[i]:
                    diff += 1
        sig_img.append(list(map(int,sig_ps_ind)))

        model = PagerankDecoder(class_weight="balanced")
        model.fit(pagerank.pr[:,sig_ps_ind],label)
        pred = model.predict(pagerank.pr[target,sig_ps_ind].reshape(1,-1))
        print(target,all_label[target],pred,diff)
        if all_label[target] == pred[0]:
            accuracy += 1
    accuracy = accuracy/51
    print(accuracy)

    plt.matshow(sig_img,aspect=20)
    plt.show()

    """
    predicted_label = []
    for i in range(len(label)):
        model = PagerankDecoder()
        predicted_label.append(model.fit_predict(PR().pr[:,sig_ps],label,i))
    predicted_label = np.array(predicted_label)
    label = np.array(label)
    print(sum(label == predicted_label)/51)
    """

    
    


