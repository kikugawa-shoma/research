import numpy as np
import os
import mne
import scipy.io

#data = scipy.io.loadmat(r"C:\Users\Owner\Downloads\Data_89\part0\003\003_task_1_.set")
data = mne.io.read_raw_eeglab(r"C:\Users\Owner\Downloads\Data_89\part0\003\003_task_1_.set",preload=True)
