import dill
import numpy as np
from dictionarylearning_calc import decode_image
import pandas as pd

dill.load_session("session.pkl")

D=dico.components_
X=dico.transform(PatchData)
Y_=np.dot(X,D)
plt.figure()
plt.imshow(decode_image(Y_[:1024,:]),cmap="gray")
plt.show(block=False)
plt.figure()
plt.imshow(decode_image(PatchData[:1024,:]),cmap="gray")
plt.show(block=False)

s1=pd.Series(PatchData[:,:].reshape(5120*64))
s2=pd.Series(Y_[:,:].reshape(5120*64))
print(s1.corr(s2))
