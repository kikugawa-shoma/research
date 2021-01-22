import scipy.io
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import accuracy_score
from collections import defaultdict

with open("results//confusion_mat_classified_label.txt") as f:
    L = list(map(int,f.readline().split()))

LL = np.load(r"results\pagerank_classified_label.npy") #ターゲット被験者が判断されたラベル
L_unique = list(set(L))

filepath = r"C:\Users\ktmks\Documents\research\tmp_results\for_python_data\brain_f_data.mat"
data = scipy.io.loadmat(filepath)
y = data["label"]
x = data["data"]

x = np.array([x[i] for i in range(len(y)) if y[i][0] == 3 or y[i][0] == 4])
y = np.array([y[i][0] for i in range(len(y)) if y[i][0] == 3 or y[i][0] == 4])

subjects_N = 51

base_train_index = defaultdict(lambda:[])
for i in range(subjects_N):
    base_train_index[L[i]].append(i)

accuracy_test = [0]*subjects_N
for i in range(subjects_N):
    target_label = LL[i]
    train_subjs = list(set(base_train_index[target_label])-set([i]))
    train_subjs.sort()
    train_ind = []
    test_ind = []
    for train_subj in train_subjs:
        train_ind.extend([80*train_subj+itr for itr in range(80)])
    test_ind = [i*80+itr for itr in range(80)]
    x_train,y_train,x_test,y_test = x[train_ind],y[train_ind],x[test_ind],y[test_ind]

    model = SVC(kernel="linear")
    model.fit(x_train,y_train)
    pred_test = model.predict(x_test)
    accuracy_test[i] = accuracy_score(y_test,pred_test)
    print(i," : ",accuracy_test[i])





f = open("C:\\Users\\ktmks\\Documents\\research\\tmp\\results\\svm_raw_pagerank_classified3.txt","w")
for i in range(subjects_N):
    print(accuracy_test[i],file=f,end=" ")
f.close()
