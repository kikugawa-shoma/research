from sklearn.svm import SVC
import numpy as np
import PageRank as PR
from collections import defaultdict
import copy
import community_find as cf
import matplotlib.pyplot as plt

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
    accuracy = 0
    sig_img = []
    predicted_label = []
    for target in range(51):
        # label = cf.ConfusionMatrix().community_detection_without(target)

        # This is the comment option
        with open(r"results\confusion_mat_classified_label.txt") as f:
            all_label = list(map(int,f.readline().split()))
        label = copy.copy(all_label)
        label[target] = None

        pagerank = PR.PageRanks(weighted=True)

        sig_ps_ind = pagerank.ttest_significant_ind(target = target,alpha=0.03,sampling=None,sample_diff=30)
        sig_img.append(sig_ps_ind)
        sig_ps_ind1 = np.load(r"results\e_num5\sig_ps.npy")

        model = PagerankDecoder(C=1,gamma="scale",class_weight="balanced")
        
        model.fit(np.delete(pagerank.pr[:,sig_ps_ind],14,0),np.delete(label,14,0))
        pred = model.predict(pagerank.pr[target,sig_ps_ind].reshape(1,-1))
        predicted_label.append(pred[0])
        print(target,all_label[target],pred,sum(sig_ps_ind))
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