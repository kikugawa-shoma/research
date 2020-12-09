import scipy.io
import numpy as np
import matplotlib.pyplot as plt
import community_find as cf
from collections import defaultdict
from scipy import stats
import itertools

class PageRanks():
    """
    rsfMRIから計算した各被験者のPagerankのクラス

    Parameters
    ----------
    filepath : ロードすべきCorrPvalue.pyでpagerankが保存された
               txtファイルへのパス。
               デフォルトは"results\featur_values.txt"

    Attributes
    ----------
    self.N  : 被験者の人数
    self.pr : 各被験者のpagerank[subject num,searchligh num]
    """
    def __init__(self,
                 filepath=r"C:\Users\ktmks\Documents\research\tmp\results\feature_values.txt"
                 ):
        subj_list = scipy.io.loadmat("C:\\Users\\ktmks\\Documents\\my_matlab\\use_subj.mat")["list"][0][:]
        self.N = len(subj_list)
        self.pr = []
        with open(filepath) as f:
            tmp = f.read().split("\n")
            for i in range(len(subj_list)):
                self.pr.append(list(map(float,tmp[i].split())))
        self.pr = np.array(self.pr)
    
    def normalize1(self):
        for i in range(self.N):
            self.pr[i] = self.pr[i]/np.linalg.norm(self.pr[i],2)
    def normalize2(self):
        for i in range(len(self.pr[0])):
            self.pr[:,i] = self.pr[:,i]/np.linalg.norm(self.pr[:,i],2)
    
    def show(self):
        """
        pagerankを画像として表示
        """
        plt.figure(figsize=[20,20])
        plt.imshow(self.pr,aspect=7)
        plt.show()
    

    def distances(self):
        """
        各被験者のpagerankベクトルの距離を計算
        """
        D = np.array([[0]*self.N for _ in range(self.N)],dtype="float32")
        for i in range(self.N):
            for j in range(self.N):
                D[i][j] = np.linalg.norm(self.pr[i]-self.pr[j])
        return D
    
    def ttest_significant_ind(self,target,alpha=0.05,sampling=None,sample_diff=5):
        """
        被験者をtargetを除いたグラフクラスタリングで分けた2群間の
        各サーチライトでのt検定を行い、p値がalpha以下のサーチライト
        のインデックスをTrue、逆をFalseとしたリストを返すメソッド


        Parameters
        ----------
        target : target subjectのindex
        alpha  : 有意水準

        Returns
        ----------
        ps<alpha : list [1,searchligh num]
        """
        # targetを除いたグラフでのグラフクラスタリングを実行
        subj_classes = cf.ConfusionMatrix().community_detection_without(target)

        with open(r"results\confusion_mat_classified_label.txt") as f:
            subj_classes = list(map(int,f.readline().split()))
        subj_classes[target] = None
        subj_classes[14] = None

        # 各被験者のpagerankをグラフクラスタリングでのsubjectのクラスに分ける
        labels = list(filter(lambda x:x is not None,set(subj_classes)))
        classified_pagerank = defaultdict(list)
        for i in range(self.N):
            if subj_classes[i] == None:
                continue
            classified_pagerank[subj_classes[i]].append(self.pr[i])
        
        # オーバーサンプリング(少ないほうの被験者クラスタのページランクをクロスオーバーにより水増しする)
        if sampling == "over":
            s_label,l_label = 0,1
            if len(classified_pagerank[s_label]) > len(classified_pagerank[l_label]):
                s_label,l_label = l_label,s_label
            for i in range(sample_diff):
                classified_pagerank[s_label].append(classified_pagerank[s_label][i%len(classified_pagerank[s_label])])

        elif sampling == "under":
            s_label,l_label = 0,1
            if len(classified_pagerank[s_label]) > len(classified_pagerank[l_label]):
                s_label,l_label = l_label,s_label
            for i in range(sample_diff):
                del classified_pagerank[l_label][0]

        # pagerankの各要素に対して2 subject class間で平均に関するt検定
        _,ps = stats.ttest_ind(classified_pagerank[labels[0]],
                                classified_pagerank[labels[1]])
        

        # 描画（デバッグ用）
        """
        plt.hist(ps)
        plt.show()
        """
        # 有意水準以下のものをTrue、それ以外をFalse
        return ps<alpha 

if __name__ == "__main__":
    pagerank = PageRanks()
    for i in range(51):
        tmp = pagerank.ttest_significant_ind(target=i)
        print(sum(tmp))






