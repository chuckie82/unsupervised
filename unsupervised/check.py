import numpy as np
import matplotlib.pyplot as plt
import h5py
f = h5py.File("6NYF_signoise40_phi0p018.h5","r")
raw = f["/MDF/images/0/image"].value
f.close()
img = np.load("/project/projectdirs/lcls/chuck/unsupervised/6NYF_meanImg.npy")
plt.subplot(121)
plt.imshow(raw); plt.title("raw image")
plt.subplot(122)
plt.imshow(img,vmin=2500); plt.title("reconstruction") 
plt.show()
