import scipy.io
from sklearn.linear_model import LogisticRegression
import numpy as np
from sklearn.metrics import accuracy_score

filepath = r"C:\Users\ktmks\Documents\research\tmp_results\for_python_data\brain_f_data.mat"
data = scipy.io.loadmat(filepath)
y = data["label"]
x = data["data"]

x = np.array([x[i] for i in range(len(y)) if y[i][0] == 3 or y[i][0] == 4])
y = np.array([y[i][0] for i in range(len(y)) if y[i][0] == 3 or y[i][0] == 4])

subjects = []
subject_N = 51

accuracy_test = [[0]*subject_N for _ in range(subject_N)]
model = LogisticRegression(max_iter=1000)

for i in range(subject_N):
    for j in range(subject_N):
        if i == j:
            continue
        train_index = [i*80+k for k in range(80)]
        test_index = [j*80+k for k in range(80)]
        x_train,y_train,x_test,y_test = x[train_index],y[train_index],x[test_index],y[test_index]
        model.fit(x_train,y_train)
        pred_test = model.predict(x_test)
        accuracy_test[i][j] = accuracy_score(y_test,pred_test)

f = open("results/svm_raw_logistice_res.txt","w")
for i in range(subject_N):
    for j in range(subject_N):
        f.write(str(accuracy_test[i][j])+" ")
    f.write("\n")
f.close()


