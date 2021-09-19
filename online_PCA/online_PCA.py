import numpy as np
import numpy.linalg as lng
import scipy as sc
import scipy.io
import time     
import random
import argparse
import os

def loguniform(low=0.1, high=1, size=None):
	return 10**(np.random.uniform(low, high))


def generate_random_signal():
	"""
	Return a fingerprint with parameter randomly drawn.
	"""
	mos = random.uniform(0.1,0.5)
	t1f = loguniform(np.log10(0.1),np.log10(6))
	t2f = loguniform(np.log10(0.01),np.log10(3))
	r = random.uniform(10,100)
	t1s = loguniform(np.log10(0.1),np.log10(6))
	t2s = loguniform(np.log10(1e-3),np.log10(1e-1))
	# We should put in what follows a command allowing to compute the fingerprint and the 
	# derivatives with respect to parameters.
	# s,ds = .......... TO BE COMPLETED
	return s[:,0]

def generate_next_batch(next_batch,B):
	for i in range(B):
		next_batch[:,i] = generate_random_signal()
	return next_batch

def block_power_MRF_online(dimension, number, usefiles=False, path='MRF/Offline/loading_data'):
	"""
	Compute 'dimension' orthonormal basis functions to minimize the distance between the manifold associated with the bloch equation and the space spanned by those 'dimension' functions.
	"""
	usefiles = True
	if usefiles:
		numfile = 1
		data = np.load(os.path.join(path,'fingerprints'+str(numfile)+'.npy'))
		ndata = data.shape[0]
		countinfile = 0
	# length of the fingerprints
	n = 666
	count = 0
	H = np.zeros((n,dimension))
	for i in range(dimension):
		H[:,i] = np.random.normal(0,1,n)
	Q,_ = np.linalg.qr(H)
	B = 4096	
	T = int(ndata // B)
	batch = np.zeros((n,B))
	
	if usefiles:
		if countinfile + B - 1 > ndata:
			numfile += 1
			data = np.load(os.path.join(path,'fingerprints'+str(numfile)+'.npy'))
			ndata = data.shape[0]
			countinfile = 0
		batch = data[countinfile:countinfile+B]
		countinfile += B
	else:
		batch = generate_next_batch(batch,B)
		
	condition = True

	while(condition):
		print('ok')
		S = np.zeros((n,dimension))
		for j in range(B):
			S += (1/B) * (np.dot(batch[j,:].reshape(-1,1),batch[j,:].reshape(1,-1).dot(Q)))
		count += 1
		Q,_ = np.linalg.qr(S)
		if usefiles:
			if countinfile + B - 1 >= ndata:
				numfile += 1
				data = np.load(os.path.join(path,'fingerprints'+str(numfile)+'.npy'))
				ndata = data.shape[0]
				countinfile = 0
			batch = data[countinfile:countinfile+B]
			countinfile += B
		else:
			batch = generate_next_batch(batch,B)
				
		if count % 10 == 0:
			sc.io.savemat('eigenvectors.mat', {'eigenvectors': Q})
			
		if usefiles:
			condition = not((numfile == number) and (countinfile+B-1 >= ndata))
		else:
			condition = count < T
	sc.io.savemat('eigenvectors.mat', {'eigenvectors': Q})
	return Q
	
if __name__ == "__main__":
	parser = argparse.ArgumentParser(description='Definition of the dimension of the subspace to approximate the manifold.')
	parser.add_argument('--dimension', type=int)
	parser.add_argument('--number', type=int)
	parser.add_argument('--usefiles', type=bool, default=True)
	parser.add_argument('--path', type=str, default='MRF/Offline/loading_data')
	args = parser.parse_args()
	block_power_MRF_online(args.dimension, args.number, usefiles = args.usefiles, path = args.path)
	
	
	
	