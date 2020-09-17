import scipy.io
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
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

accuracy_test = [0]*subject_N
model = SVC(kernel="linear")

for i in range(subject_N):  
    test_index = [i*80+k for k in range(80)]
    train_index = list(set([k for k in range(4080)]) - set(test_index))
    x_train,y_train,x_test,y_test = x[train_index],y[train_index],x[test_index],y[test_index]
    model.fit(x_train,y_train)
    pred_test = model.predict(x_test)
    accuracy_test[i] = accuracy_score(y_test,pred_test)
    print(i," : ",accuracy_test[i])

f = open("C:\\Users\\ktmks\\Documents\\research\\tmp\\results\\svm_raw.txt","w")
for i in range(subject_N):
    print(accuracy_test[i],file=f,end=" ")
f.close()