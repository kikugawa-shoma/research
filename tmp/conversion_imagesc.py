import numpy as np
import matplotlib.pyplot as plt

f = open("results/svm_raw_conversion_res.txt","r")
tmp = f.read().split("\n")
conv_mat = [list(map(float,t.split())) for t in tmp]
del conv_mat[-1]
#conv_mat = np.array(conv_mat)

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
im = ax.imshow(conv_mat)
fig.colorbar(im)
plt.show(True)





