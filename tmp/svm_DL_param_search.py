import scipy.io
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.model_selection import LeaveOneGroupOut
import numpy as np
from sklearn.metrics import accuracy_score
import tqdm


filepath = r"C:\Users\Owner\ishii_lab\data\time_search\omp\res_AtomN-1000_SparseDegree-100_MaxIter-1000_dim-3530_sample_num-15504alpha1.mat"

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

parameters = [0.04,0.2,1,5,25,125,625,3125]


accuracy_test = [[0]*subject_N for i in range(len(parameters))]


for j in tqdm.tqdm(range(len(parameters))):
    i = 0
    for train_index, test_index in tqdm.tqdm(logo.split(x,y,subjects),leave=False):
        x_train,x_test = x[train_index],x[test_index]
        y_train,y_test = y[train_index],y[test_index]
        model = SVC(kernel="rbf",C=parameters[j],gamma="scale")
        model.fit(x_train,y_train)

        pred_test = model.predict(x_test)
        accuracy_test[j][i] = accuracy_score(y_test,pred_test)
        i += 1

f = open("svm_DL_param_search_res.txt","x")
for j in range(len(parameters)):
    f.write("C : "+str(parameters[j])+"  ")
    for i in range(len(accuracy_test[0])):
        f.write(str(accuracy_test[j][i])+" ")
    f.write("\n")
f.close()


