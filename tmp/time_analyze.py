import scipy.io
import matplotlib.pyplot as plt

dims=[i for i in range(100,1001,100)]
sample_nums=[100,200,300,500,600,700,800,900,1000,2000,3000,4000,5000,6000,7000]
dim_t=[0]*len(dims)
sample_t=[0]*len(sample_nums)

for i in range(len(dims)):
    filepath=r"C:\Users\Owner\ishii_lab\data\time_search\omp\r"
    dim = dims[i]
    sample_num = 1000
    filename="es_AtomN-100_SparseDegree-10_MaxIter-1000_dim-"+str(dim)+"_sample_num-"+str(sample_num)+".mat"
    filepath=filepath+filename
    data=scipy.io.loadmat(filepath)
    dim_t[i] = data["t"][0][0]

for i in range(len(sample_nums)):
    filepath=r"C:\Users\Owner\ishii_lab\data\time_search\omp\r"
    dim = 100
    sample_num = sample_nums[i]
    filename="es_AtomN-100_SparseDegree-10_MaxIter-1000_dim-"+str(dim)+"_sample_num-"+str(sample_num)+".mat"
    filepath=filepath+filename
    data=scipy.io.loadmat(filepath)
    sample_t[i] = data["t"][0][0]

print(sample_t)
print(dim_t)
plt.plot(dims,dim_t)
plt.show()

plt.plot(sample_nums,sample_t)
plt.show()