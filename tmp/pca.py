def feature_value_pca(X,n_c):
    '''
    Xをpcaにより成分数n_cで次元削減する関数
    '''
    from sklearn.decomposition import PCA
    import scipy

    X = scipy.stats.zscore(X)
    pca = PCA(n_c)
    pca.fit(X)
    X = pca.transform(X)
    return X


if __name__ == "__main__":
    '''
    pagerankのデータを成分数2個でpcaにより次元削減をしたのちプロットするスクリプト
    '''

    import scipy.io
    import statistics as st
    import numpy as np
    import matplotlib.pyplot as plt

    subj_list = scipy.io.loadmat("C:\\Users\\ktmks\\Documents\\my_matlab\\use_subj.mat")["list"][0][:]
    data_root = "C:\\Users\\ktmks\\Documents\\my_sources\\20\\"

    T = []

    f = open(r"C:\Users\ktmks\Documents\research\tmp\results\feature_values.txt",mode="r")
    tmp = f.read().split("\n")
    for i in range(len(subj_list)):
        T.append(list(map(float,tmp[i].split())))
    f.close()
    T = np.array(T)
    pca_T = feature_value_pca(T,2)
    
    f = open(r"C:\Users\ktmks\Documents\research\tmp\results\confusion_mat_classified_label.txt")
    L = list(map(int,f.readline().split()))
    c = []
    for i in L:
        if i == 0:
            c.append("blue")
        else:
            c.append("red")

    plt.scatter(T[:,0],T[:,1],c=c)
    plt.show()

