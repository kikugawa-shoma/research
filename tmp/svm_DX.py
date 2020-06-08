import scipy.io
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import accuracy_score


filepath = r"C:\Users\Owner\ishii_lab\data\time_search\omp\res_AtomN-100_SparseDegree-10_MaxIter-1000_dim-3530_sample_num-15504alpha1.mat"

data = scipy.io.loadmat(filepath)

y = data["label"]
x = data["Y_"]

x = [x[i] for i in range(len(y)) if y[i][0] == 3 or y[i][0] == 4]
y = [y[i][0] for i in range(len(y)) if y[i][0] == 3 or y[i][0] == 4]

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.3)

model = SVC(kernel="linear")

model.fit(x_train,y_train)

pred_test = model.predict(x_test)
accuracy_test = accuracy_score(y_test,pred_test)
print(accuracy_test)

