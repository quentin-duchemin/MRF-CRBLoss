import sys
import scipy as sc
import os
import numpy as np

folder ='v3_uniform_b1cutoff_B0B1vary_R13_complex'
savefolder = 'v3_uniform_b1cutoff_B0B1vary_R13_complex_1024'

path = '/gpfs/data/asslaenderlab/share/zhangx19/code-MRF-april20/MRF/Offline/loading_data/'
save_path = os.path.join(path, savefolder)
data_path = os.path.join(path, folder)
count = 1
for num in range(1,1200,2):
    try:
        print(num)
        data1 = np.load(data_path+'/fingerprints'+str(num)+'.npy')
        params1 = np.load(data_path+'/params'+str(num)+'.npy')
        CRBs1 = np.load(data_path+'/CRBs'+str(num)+'.npy')

        data2 = np.load(data_path + '/fingerprints' + str(num+1) + '.npy')
        params2 = np.load(data_path + '/params' + str(num+1) + '.npy')
        CRBs2 = np.load(data_path + '/CRBs' + str(num+1) + '.npy')

        fingerprints = np.concatenate((data1, data2), axis=0)
        params = np.concatenate((params1, params2), axis=0)
        CRBs = np.concatenate((CRBs1, CRBs2), axis=0)

        np.save(save_path + '/params' + str(count) + '.npy', params)
        np.save(save_path + '/CRBs' + str(count) + '.npy', CRBs)
        np.save(save_path + '/fingerprints' + str(count) + '.npy', fingerprints)
        count = count+1
    except:
        continue