import numpy as np
import matplotlib.pyplot as plt

def matrix_swapper(M):
    with open(r"C:\Users\ktmks\Documents\research\tmp\results\confusion_mat_classified_label.txt") as f:
        L = list(map(int,f.readline().split()))
        L = np.argsort(L)
    M = M[:,L]
    M = M[L,:]
    return M


def confusion_show(filepaths):
    for cnt,filepath in enumerate(filepaths):
        with open(filepath,mode="r") as f:
            tmp = f.read()
        tmp = tmp.rsplit("\n")
        acc = []
        for f_row in tmp:
            if f_row:
                acc.append(list(map(float,f_row.split())))
        acc = np.array(acc)
        acc = matrix_swapper(acc)
        plt.subplot(2,2,2*cnt+1)
        plt.imshow(acc)
        acc = acc>0.6
        plt.subplot(2,2,2*cnt+2)
        plt.imshow(acc)


    plt.show()


if __name__ == "__main__":
    filepaths = [r"C:\Users\ktmks\Documents\research\tmp\results\svm_raw_logistice_res.txt",
                 r"C:\Users\ktmks\Documents\research\tmp\results\svm_raw_conversion_res.txt"]
    confusion_show(filepaths)
