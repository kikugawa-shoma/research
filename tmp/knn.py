import scipy.io
from sklearn.svm import SVC
import numpy as np
from sklearn.metrics import accuracy_score
import PageRank as PR
from statistics import mean
from matplotlib import pyplot as plt


subj_label = np.load(r"results\pagerank_classified_label.npy")

subj_N = len(subj_label)

filepath = r"C:\Users\ktmks\Documents\research\tmp_results\for_python_data\brain_f_data.mat"
data = scipy.io.loadmat(filepath)
y = data["label"]
x = data["data"]
x = np.array([x[i] for i in range(len(y)) if y[i][0] == 3 or y[i][0] == 4])
y = np.array([y[i][0] for i in range(len(y)) if y[i][0] == 3 or y[i][0] == 4])

pageranks = PR.PageRanks(feature="dmap")


distance = pageranks.distances()


def knn_decoding(K=49,weighted=False):
    accuracies = [0]*subj_N
    for i in range(subj_N):

        # cos類似度の高い被験者k人を選ぶ
        distance[i][i] = -10
        sort_ind = np.argsort(distance[i])[::-1]
        selected_ind = sort_ind[:K]


        # k個のSVMを訓練する
        models = []
        for k in range(K):
            models.append(SVC())
        for k in range(K):
            x_train = x[selected_ind[k]*80:(selected_ind[k]+1)*80,:]
            y_train = y[selected_ind[k]*80:(selected_ind[k]+1)*80]
            models[k].fit(x_train,y_train)


        # k個のSVMの投票によりターゲットの注意方向を推定
        x_test = x[i*80:(i+1)*80,:]
        y_test = y[i*80:(i+1)*80]
        preds = []
        for k in range(K):
            preds.append(models[k].predict(x_test))
        pred = [0]*len(y_test)
        for j in range(len(y_test)):
            vote3,vote3_,vote4,vote4_ = 0,0,0,0
            for k in range(K):
                if preds[k][j] == 3:
                    if weighted:
                        vote3 += 1*distance[i][selected_ind[k]]
                        vote3_ += 1
                    else:
                        vote3 += 1
                if preds[k][j] == 4:
                    if weighted:
                        vote4 += 1*distance[i][selected_ind[k]]
                        vote4_ += 1
                    else:
                        vote4 += 1
            if vote3>vote4:
                pred[j] = 3
            else:
                pred[j] = 4
        accuracies[i] = accuracy_score(y_test,pred)
    return accuracies
    

accs = []
for i in range(50):
    accuracy = knn_decoding(K=i)
    print(mean(accuracy))
    accs.append(mean(accuracy))

plt.plot([i for i in range(len(accs))],accs)
plt.show()

#a = knn_decoding(49,weighted=True)

