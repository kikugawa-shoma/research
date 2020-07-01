from sklearn.decomposition import DictionaryLearning
import scipy.io
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd

def decode_image(image):
    decoded_image=np.empty((256,256))
    for i in range(1024):
        r=i%32
        c=i//32
        decoded_image[r*8:r*8+8,c*8:c*8+8]=image[i].reshape([8,8],order="F")
    return decoded_image


if __name__ == "__main__":

    MatPatchedImage=scipy.io.loadmat(r"C:\Users\ktmks\Documents\Dic_ler1\ver1_02\mono\PatchData.mat")
    PatchData=np.array(MatPatchedImage["PatchData"]).T

    decoded_image=decode_image(PatchData[:1024,:])

    dico=DictionaryLearning(n_components               =    128,
                            transform_n_nonzero_coefs  =      8,
                            verbose                    =   True,
                            max_iter                   =   1000,
                            )
    Dict=dico.fit(PatchData)
    print("Hello")

