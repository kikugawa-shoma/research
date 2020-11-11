from sklearn.svm import SVC
import numpy as np
from svm_raw_nearests_decoding import PageRanks as PR
from collections import defaultdict
import copy

class PagerankDecoder(SVC):
    def __init__(self,kernel="rbf"):
        super().__init__(kernel=kernel)
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

def t_test_classifier(pagerank,label,test_ind):
    from scipy import stats
    N = len(label)
    train_ind = [True]*N
    train_ind[test_ind] = False
    pagerank = np.array(pagerank)
    label = np.array(label)

    train_x = pagerank[train_ind]
    train_y = label[train_ind]
    c_pagerank = [[] for _ in range(len(set(label)))]
    for i in range(len(train_y)):
        c_pagerank[train_y[i]].append(train_x[i])
    sum_ps = []
    for i in range(2):
        cc_pagerank = copy.deepcopy(c_pagerank)
        cc_pagerank[i].append(pagerank[test_ind])
        ts,ps = stats.ttest_ind(cc_pagerank[0],cc_pagerank[1])
        sum_ps.append(sum(ps))
    return sum_ps.index(min(sum_ps))
    

if __name__ == "__main__":
    with open(r"results\confusion_mat_classified_label.txt") as f:
        label = list(map(int,f.readline().split()))
    
    ps = np.load(r"results\pagerank_p-value.npy")
    sig_ps = ps<0.05
    predicted_label = []
    for i in range(len(label)):
        model = PagerankDecoder()
        predicted_label.append(model.fit_predict(PR().pr[:,sig_ps],label,i))
    predicted_label = np.array(predicted_label)
    label = np.array(label)
    print(sum(label == predicted_label)/51)

    """
    predicted_label = []
    for i in range(len(label)):
        predicted_label.append(t_test_classifier(PR().pr,label,i))
    label = np.array(label)
    predicted_label = np.array(predicted_label)
    print(sum(label == predicted_label)/51)
    """
    
    


