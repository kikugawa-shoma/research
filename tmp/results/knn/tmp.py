import numpy as np
from matplotlib import pyplot as plt
"""
k-nearest neighbor で被験者をk人選び、それぞれのSVMを訓練し、
それらのSVMの投票によりターゲットのデコーディングを行った結果を
グラフにして表示するスクリプト
"""

def make_plot(filename):
    data = np.load(filename)
    fig = plt.figure()
    plt.plot([i for i in range(len(data))],data)
    plt.title(filename)
    plt.show()
    fig.savefig("C:\\Users\\ktmks\\OneDrive\\画像\\tmp\\"+filename+".png")

filenames = [
"dmap_with_t-test.npy",
"dmap_without_t-test.npy",
"pagerank_with_t-test.npy",
"pagerank_without_t-test.npy",
]

for i in range(len(filenames)):
    make_plot(filenames[i])
