import scipy.io
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.model_selection import LeaveOneGroupOut
import numpy as np
from sklearn.metrics import accuracy_score


filepath = r"C:\Users\Owner\ishii_lab\data\time_search\omp\res_AtomN-1000_SparseDegree-100_MaxIter-1000_dim-3530_sample_num-15504alpha1.mat"

#minibatch
#filepath = r"C:\Users\Owner\ishii_lab\data\minibatch\omp\minibatch_res_AtomN-1000_SparseDegree-100_MaxIter-10000_dim-3530_sample_num-15504_alpha-1_batchsize-3.mat"

data = scipy.io.loadmat(filepath)

y = data["label"]
x = data["X"]

x = np.array([x[i] for i in range(len(y)) if y[i][0] == 3 or y[i][0] == 4])
y = np.array([y[i][0] for i in range(len(y)) if y[i][0] == 3 or y[i][0] == 4])


subjects = []
subject_N = 51
for i in range(subject_N):
    for j in range(80):
        subjects.append(i)

logo =LeaveOneGroupOut()
logo.get_n_splits(x,y,subjects)
accuracy_test = [0]*subject_N

i = 0
for train_index, test_index in logo.split(x,y,subjects):
    x_train,x_test = x[train_index],x[test_index]
    y_train,y_test = y[train_index],y[test_index]
    model = SVC(kernel="rbf")
    model.fit(x_train,y_train)

    pred_test = model.predict(x_test)
    accuracy_test[i] = accuracy_score(y_test,pred_test)
    i += 1
    print(accuracy_test)

