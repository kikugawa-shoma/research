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
        self.T = T #pagerank
        self.N = T.shape[0]
    
    def thleashold_fit(self,subj_ind,thleashold):
        T_argsort = self.T[subj_ind].argsort()
        train_subjs=[]
        for i in range(1,self.N):
            if self.T[subj_ind][T_argsort[i]] < thleashold:
                train_subjs.append(T_argsort[i])

        inds = []
        for i in range(len(train_subjs)):
            ind = train_subjs[i]
            inds.extend([ind*80+j for j in range(80)])
            
        x_train = self.X[inds]
        y_train = self.Y[inds]
        super().fit(x_train,y_train)
    
    def predict(self,x_test):
        return super().predict(x_test)

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

    thleasholds = [0.040+i*0.001 for i in range(2)]
    accuracy_test = [[0]*subject_N for _ in range(len(thleasholds))]

    for j,tl in enumerate(thleasholds):
        for i in range(subject_N):
            model = NearestsDecoder(x,y,D,kernel="linear")
            model.thleashold_fit(subj_ind=i,thleashold=tl)
            x_test = x[80*i:80*(i+1),:]
            y_test = y[80*i:80*(i+1)]
            pred_test = model.predict(x_test)
            accuracy_test[j][i] = accuracy_score(y_test,pred_test)
            print(i," : ",accuracy_test[j][i])
    np.save("results/svm_raw_nearests_decoding_enum5",accuracy_test)

    



