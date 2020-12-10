from sklearn.svm import SVC
import numpy as np
import PageRank as PR
from collections import defaultdict
import copy
import community_find as cf
import matplotlib.pyplot as plt
from statistics import mean

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
        label = np.array(label)

        pagerank = PR.PageRanks()

        sig_ps_ind = pagerank.ttest_significant_ind(target = target,alpha=0.005,sampling="over",sample_diff=30)
        sig_img.append(sig_ps_ind)
        sig_ps_ind1 = np.load(r"results\e_num5\sig_ps.npy")

        model = PagerankDecoder(C=1,gamma="scale",class_weight="balanced")
        
        model.fit(np.delete(pagerank.pr[:,sig_ps_ind],14,0),np.delete(label,14,0))
        pred = model.predict(pagerank.pr[target,sig_ps_ind].reshape(1,-1))
        predicted_label.append(pred[0])
        if all_label[target] == pred[0]:
            accuracy += 1

        # analyze
        if target == 14:
            continue
        sig_ps_indexes = [i for i,x in enumerate(sig_ps_ind) if x]
        tmp = 0
        for i in range(len(sig_ps_indexes)):
            m_other = mean(pagerank.pr[label==(1-all_label[target]),sig_ps_indexes[i]])
            m_target = mean(pagerank.pr[label==(all_label[target]),sig_ps_indexes[i]])

            if abs(pagerank.pr[target,sig_ps_indexes[i]]-m_target)<abs(pagerank.pr[target,sig_ps_indexes[i]]-m_other):
                tmp += 1
        print("{} {} {} {} {}".format(target,all_label[target],pred[0],sum(sig_ps_ind),tmp/len(sig_ps_indexes)))
        
    accuracy = accuracy/51
    print(accuracy)
    plt.matshow(sig_img,aspect=20)
    plt.show()




