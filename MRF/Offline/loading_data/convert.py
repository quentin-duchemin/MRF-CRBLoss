import numpy as np
import os
import scipy as sc
import scipy.io

for i in range(351):
	a = sc.io.loadmat('fingerprints'+str(i)+'.mat')['fingerprints']
	np.save('fingerprints'+str(i)+'.npy',a)
	a = sc.io.loadmat('CRBs'+str(i)+'.mat')['CRBs']
	np.save('CRBs'+str(i)+'.npy',a)
	a = sc.io.loadmat('params'+str(i)+'.mat')['params']
	np.save('params'+str(i)+'.npy',a)
