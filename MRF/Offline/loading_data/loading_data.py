import time
import scipy as sc
from scipy import io
import numpy as np
import argparse
import os


if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Loading data.')
	# parser.add_argument('--urls_file', type=str)
	parser.add_argument('--data_path_root', type=str, default='/scratch/kl3141/MRF/code-MRF-april20/MRF/Offline/loading_data/m0sT1')
	parser.add_argument('--save_name', type=str)
	args = parser.parse_args()
	
	
	if not os.path.exists(args.save_name):
		os.mkdir(args.save_name)
	t0 = time.time()
	num = 1

	data_path_root = args.data_path_root
	data_paths = sorted(os.listdir(data_path_root))


	num_files = int(len(data_paths)/3)

	for file_num in range(1,num_files+1):
		print(num)
		data_fingerprinting = sc.io.loadmat(os.path.join(data_path_root, 'fingerprints'+str(file_num)+'.mat'))
		data_params = sc.io.loadmat(os.path.join(data_path_root, 'params' + str(file_num) + '.mat'))
		data_CRBs = sc.io.loadmat(os.path.join(data_path_root, 'CRBs' + str(file_num) + '.mat'))


		np.save(args.save_name + '/fingerprints' + str(num) + '.npy', np.array(data_fingerprinting['fingerprints']))
		np.save(args.save_name + '/params' + str(num) + '.npy', np.array(data_params['params']))
		np.save(args.save_name + '/CRBs' + str(num) + '.npy',  np.array(data_CRBs['CRBs']))
		# If the files contain the CRBs
		# CRBs = data['CRB_all'].T
		# Order the CRB vector in order to respect the numerotation 0=m0s, 1=T1, 2=T2f, 3=R, 4=T2s, 5=PD.
		# CRBs = CRBs[:, [1, 2, 3, 4, 5, 0]] already did that during signal generation
		num += 1
		
	print(time.time()-t0)
