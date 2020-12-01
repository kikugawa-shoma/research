import scipy.io
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import accuracy_score
from collections import defaultdict
import scipy.io

f = open("C:\\Users\\ktmks\\Documents\\research\\tmp\\results\\confusion_mat_classified_label.txt")
L = list(map(int,f.readline().split()))
f.close()
L = scipy.io.loadmat(r"C:\Users\ktmks\Documents\my_matlab\make_figures\confusion_mat_classified_label_kmeans.mat")["label1"]
L = [l[0] for l in L]
L_unique = list(set(L))

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

labeled_x = defaultdict(lambda :[])
labeled_y = defaultdict(lambda :[])

for i in range(subject_N):
    Li = L[i]
    labeled_x[Li] += x[80*i:80*(i+1),:].tolist()
    labeled_y[Li] += y[80*i:80*(i+1)].tolist()

for key in L_unique:
    labeled_x[key] = np.array(labeled_x[key])
    labeled_y[key] = np.array(labeled_y[key])

label_cnt = [0]*len(L_unique)

for i in range(subject_N):
    if i == 14:
        accuracy_test[i] = 0.5
        continue

    Li = L[i]

    test_index = [label_cnt[Li]*80+k for k in range(80)]
    train_index = list(set([k for k in range(len(labeled_x[Li]))]) - set(test_index))
    x_train,y_train,x_test,y_test = labeled_x[Li][train_index],labeled_y[Li][train_index],labeled_x[Li][test_index],labeled_y[Li][test_index]
    model.fit(x_train,y_train)
    pred_test = model.predict(x_test)
    accuracy_test[i] = accuracy_score(y_test,pred_test)
    print(i," : ",accuracy_test[i])
    label_cnt[Li] += 1

f = open("C:\\Users\\ktmks\\Documents\\research\\tmp\\results\\svm_raw_confusion_mat_kmean_classified.txt","w")
for i in range(subject_N):
    print(accuracy_test[i],file=f,end=" ")
f.close()
