import numpy as np
from matplotlib import pyplot as plt

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
