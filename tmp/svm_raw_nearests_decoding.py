import scipy.io
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

class PageRanks():
    def __init__(self,
                 filepath=r"C:\Users\ktmks\Documents\research\tmp\results\feature_values.txt"):
        subj_list = scipy.io.loadmat("C:\\Users\\ktmks\\Documents\\my_matlab\\use_subj.mat")["list"][0][:]
        self.N = len(subj_list)
        self.pr = []
        with open(filepath) as f:
            tmp = f.read().split("\n")
            for i in range(len(subj_list)):
                self.pr.append(list(map(float,tmp[i].split())))
        self.pr = np.array(self.pr)
    
    def show(self):
        plt.figure(figsize=[20,20])
        plt.imshow(self.pr,aspect=7)
        plt.show()
    

    def distances(self):
        D = np.array([[0]*self.N for _ in range(self.N)],dtype="float32")
        for i in range(self.N):
            for j in range(self.N):
                D[i][j] = np.linalg.norm(self.pr[i]-self.pr[j])
        return D

class NearestsDecoder(SVC):
    def __init__(self,X,Y,T,kernel="linear"):
        super().__init__(kernel=kernel)
        self.X = X
        self.Y = Y
        self.T = T
        self.N = T.shape
    
    def nearest_fit(self,subj_ind,train_num=10):
        T_argsort = self.T[subj_ind].argsort()
        train_subj = T_argsort[1:train_num]
        inds = []
        for i in range(train_num-1):
            ind = train_subj[i]
            inds.extend([ind*80+j for j in range(80)])
            
        x_train = self.X[inds]
        y_train = self.Y[inds]
        super().fit(x_train,y_train)
    
    def nearests_predict(self,x_test):
        return super().predict(x_test)

class SelectiveDecoder(SVC):
    def __init__(self,X,Y,T,kernel="linear"):
        super().__init__(kernel="linear")
        self.X = X
        self.Y = Y
        self.T = T
        with open(r"results\confusion_mat_classified_label.txt") as f:
            self.C = list(map(int,f.readline().split()))
    def selective_fit(self,subj_ind):
        pass



if __name__ == "__main__":
    subject_N = 51

    prs = PageRanks()
    D = prs.distances()

    filepath = r"C:\Users\ktmks\Documents\research\tmp_results\for_python_data\brain_f_data.mat"
    data = scipy.io.loadmat(filepath)

    y = data["label"]
    x = data["data"]
    x = np.array([x[i] for i in range(len(y)) if y[i][0] == 3 or y[i][0] == 4])
    y = np.array([y[i][0] for i in range(len(y)) if y[i][0] == 3 or y[i][0] == 4])

    train_num_list = [5*i for i in range(1,11)]
    accuracy_test = [[0]*subject_N for _ in range(len(train_num_list))]

    model = SelectiveDecoder(x,y,prs.pr,kernel="linear")

    """

    for j,train_num in enumerate(train_num_list):
        for i in range(subject_N):
            model = SelectiveDecoder(x,y,D,kernel="linear")
            model.nearest_fit(subj_ind=i,train_num=train_num)
            x_test = x[80*i:80*(i+1),:]
            y_test = y[80*i:80*(i+1)]
            pred_test = model.nearests_predict(x_test)
            accuracy_test[j][i] = accuracy_score(y_test,pred_test)
            print(i," : ",accuracy_test[j][i])
    np.save("results/svm_raw_nearests_decoding_enum5",accuracy_test)
    """

    



