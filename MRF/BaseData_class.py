from __future__ import print_function, division
import os
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import random
import sys
import scipy as sc
import os
from .Training_parameters import *
import time
from torch.utils.data import Dataset
from torch import optim
import scipy.integrate
import copy
from .simulate_signal import simulation

def load_data_6params():
  my_path = os.path.abspath(os.path.dirname(__file__))
  path = os.path.join(my_path, "simulate_signal/OCT_resutl_MT_100_3s_low_SAR.mat")
  x = sc.io.loadmat(path)['x']
  x[1,:] = x[1,:] * 1e-3 # convert ms to s
  return x

def loguniform(low=0.1, high=1, size=None):
    return 10**(np.random.uniform(low, high, size))
    
class BaseData_class(Dataset):
	"""
	Children classes define the way the fingerprints are generated. Pro-processing tasks (adding noise or normalization) are also handled by this class.
	"""
	def __init__(self, training_parameters, noise_type, noise_level, minPD, maxPD, CRBrequired=False):
		self.trparas     = training_parameters
		self.noise_type  = noise_type
		self.noise_level = noise_level
		self.minPD 		  = minPD
		self.maxPD       = maxPD
		self.x           = load_data_6params()
		self.TR          = TR  # defined in Training_parameters.py
		self.t           = TR * np.array([i for i in range(666)])  
		self.CRBrequired = CRBrequired
 
				
				
		# parasCRB select the relevant variable to compute the Cramer Rao Bound in the method 'compute_CRBs'.
		if self.maxPD > self.minPD:
			self.parasCRB = [0,1,2,3,4,5]
		else:
			self.parasCRB = [0,1,2,3,4]

		
	def sample(self):
		raise NotImplementedError("This method sample has to be implemented in the class inheriting from BaseData in an Online framework.")

	def dico_save(self):
		""" Save the parameters of the instance of the class BaseData_class."""
		dic = (self.__dict__).copy()
		del dic['trparas']
		dic.update(self.trparas.dico_save())
		return dic
		
	def compute_CRBs(self, ytemp, PD, nbnoises=100):
		"""
		Compute the Cramer Rao Bound for the parameters specified by the vector 'weights'.
		"""
		# I select the parameters PD, m0s, T1, T2, R and T2s (the parameter T1s is not selected). PD is selected only if it is not considered constant in our experiments.
		y = copy.deepcopy(ytemp)
		y[:,:5] *= PD #i.e. if the derivative wrt the proton density has not to be multiply by the proton density
			
		if self.noise_type == 'Standard':
			I = np.dot(y[:,self.parasCRB].T,y[:,self.parasCRB]) 
			Im1 = np.linalg.inv(I)
		    # Optimize for the average of all parameters
			C = Im1.diagonal()
			C = C[self.trparas.params]
		    # Normalize the cost; the abs is just in case the inversion failed
			C = abs(C)
		return C / nbnoises
		
		
	def proton_density_scaling(self, vector, PD=None):
		"""
		Rescale the fingerprints give nas an input by proton densities. 
		"""
		vec = copy.deepcopy(vector)
		vec = np.squeeze(vec)
		if vec.ndim > 1:
			if PD is None:
				PD = np.random.uniform(self.minPD,self.maxPD,vec.shape[0])
			vec *= np.tile(PD.reshape(-1,1),(1,vec.shape[1]))
		else:
			if PD is None:
				PD = np.random.uniform(self.minPD,self.maxPD)
			vec *= PD 
		return vec, PD


	def proton_density_scaling_B0(self, vector, PD=None):
		"""
		Rescale the fingerprints give nas an input by proton densities.
		"""
		vec = copy.deepcopy(vector)
		vec = np.squeeze(vec)
		if vec.ndim > 1:
			if PD is None:
				PD = np.random.uniform(self.minPD, self.maxPD, vec.shape[0])
			vec *= np.tile(PD.reshape((-1, 1,1)), (1, vec.shape[1], vec.shape[2]))
		else:
			if PD is None:
				PD = np.random.uniform(self.minPD, self.maxPD)
			vec *= PD
		return vec, PD


	# def proton_density_scaling_B0_complex(self, vector, PD=None):
	# 	"""
	# 	Rescale the fingerprints give nas an input by proton densities.
	# 	"""
	# 	vec = copy.deepcopy(vector)
	# 	vec = np.squeeze(vec)  # (b, timepoints, 2)
	# 	if vec.ndim > 1:
	# 		PD_real = np.random.uniform(self.minPD, self.maxPD, vec.shape[0])
	# 		PD_imag = np.random.uniform(self.minPD, self.maxPD, vec.shape[0])
	# 		PD = np.stack([PD_real,PD_imag],1) # (b,2)
	#
	#
	# 		PD_real = np.tile(PD_real.reshape((-1, 1)), (1, vec.shape[1]))
	#
	# 		PD_imag = np.tile(PD_imag.reshape((-1, 1)), (1, vec.shape[1]))
	#
	# 		vec_real = vec[:,:,0] * PD_real - vec[:,:,1]*PD_imag
	# 		vec_imag = vec[:, :, 1] * PD_real + vec[:, :, 0] * PD_imag
	# 		vec = np.stack([vec_real, vec_imag], axis=2)   # (b, timepoints, 2)
	#
	#
	#
	# 		# vec *= np.tile(PD.reshape((-1, 1,1)), (1, vec.shape[1], vec.shape[2]))
	# 	else:
	# 		if PD is None:
	# 			PD = np.random.uniform(self.minPD, self.maxPD)
	# 		vec *= PD
	# 	return vec, PD

	def proton_density_scaling_B0_complex(self, vector, PD=None):
		"""
		Rescale the fingerprints give nas an input by proton densities.
		"""
		vec = copy.deepcopy(vector)
		vec = np.squeeze(vec)  # (b, timepoints, 2)
		if vec.ndim > 1:
			# PD_real = np.random.uniform(self.minPD, self.maxPD, vec.shape[0])
			# PD_imag = np.random.uniform(self.minPD, self.maxPD, vec.shape[0])
			# PD = np.stack([PD_real,PD_imag],1) # (b,2)
			PD =  (np.random.uniform(0,1,vec.shape[0]) * 0.9 + 0.1) * np.exp(1j * 2 * np.pi * np.random.uniform(0,1,vec.shape[0]))
			PD_real = PD.real
			PD_imag = PD.imag
			PD = np.stack([PD_real,PD_imag],1) # (b,2)
			PD_real = np.tile(PD_real.reshape((-1, 1)), (1, vec.shape[1]))
			PD_imag = np.tile(PD_imag.reshape((-1, 1)), (1, vec.shape[1]))

			if len(vec.shape)==3:
				vec_real = vec[:,:,0] * PD_real - vec[:,:,1]*PD_imag
				vec_imag = vec[:, :, 1] * PD_real + vec[:, :, 0] * PD_imag
				vec = np.stack([vec_real, vec_imag], axis=2)   # (b, timepoints, 2)
			else:
				vec_real = vec * PD_real
				vec_imag = vec * PD_imag
				vec = np.stack([vec_real, vec_imag], axis=2)   # (b, timepoints, 2)

			# vec *= np.tile(PD.reshape((-1, 1,1)), (1, vec.shape[1], vec.shape[2]))
		else:
			if PD is None:
				PD = np.random.uniform(self.minPD, self.maxPD)
			vec *= PD
		return vec, PD

		
	def add_noise(self,fing):
		"""
		Add noise to a given fingerprint and perform normalization if asked. If a proton density different from 1 is used, the scaling of the fingerprint
		is also done here before adding the noise realization. 
		"""
		fingerprint = copy.deepcopy(fing)
		l = len(fingerprint)
		np.random.seed()	
		if self.noise_type == 'SNR':
			noise = np.random.normal(0, 1, l)
			signal_Power = np.linalg.norm(fingerprint)
			noise_Power = np.linalg.norm(noise)
			cst = signal_Power / (noise_Power * self.noise_level)
			noise = noise * cst
		elif self.noise_type == 'Standard':
			noise = np.random.normal(0, self.noise_level, l)
		fingerprint += noise
		if self.trparas.normalization == 'Noisy_input':
			return fingerprint / np.linalg.norm(fingerprint)
		else:
			return fingerprint
	    	
	def add_noise_batch(self,fingerprints):
		"""
		Add noise to a given batch of fingerprints and perform normalization if asked.
		If a proton density different from 1 is used, the scaling of the fingerprint
		is also done here before adding the noise realization. 
		"""
		n,l = fingerprints.shape
		np.random.seed()
		if self.noise_type == 'SNR':
			noise = np.random.normal(0, 1, (n,l))
			signal_Power = np.linalg.norm(fingerprints, axis=1)
			noise_Power = np.linalg.norm(noise,axis=1)
			cst = signal_Power / (noise_Power * self.noise_level)
			noise = noise * np.tile(cst.reshape(-1,1),(1,l))
		elif self.noise_type == 'Standard':
			noise = np.random.normal(0, self.noise_level, (n,l))
		fingerprints += noise
		if self.trparas.normalization == 'Noisy_input':
			return fingerprints / np.tile(np.linalg.norm(fingerprints,axis=1).reshape(-1,1), (1,l))
		else:
			return fingerprints

	def add_noise_batch_B0(self,fingerprints):
		"""
		Add noise to a given batch of fingerprints and perform normalization if asked.
		If a proton density different from 1 is used, the scaling of the fingerprint
		is also done here before adding the noise realization.
		"""
		n,l,r = fingerprints.shape
		np.random.seed()

		# noise_level = 0.002
		noise_level = self.noise_level
		# add noise to real and imag part seperately with different noise level
		noise_real = np.random.normal(0, noise_level, (n, l))
		noise_imag = np.random.normal(0, noise_level, (n, l))
		noise = np.stack([noise_real,noise_imag],axis=2)  # n,l,r  4000,666,2
		fingerprints += noise
		return fingerprints



	def nlls(self, fing, bounds, path, PD=1, nbnoise=5, save_name='-1'):
		"""Return the mean and the std of the estimated parameters for a specific fingerprint using Non Linear Least Squares for different noise realizations."""
		fingerprint = copy.deepcopy(fing)
		def simu(x,para0,para1,para2,para3,para4,para5):
			y,_ = simulation.simulate_MT_ODE(self.x, TR, self.t, para0, para1, para2, para3, para1, para4)
			return( para5 * y[:,0] )
		signals = np.zeros((nbnoise,len(fingerprint)))
		paras = np.zeros((nbnoise,6))
		for i in range(nbnoise):
			signals[i,:], PD = self.proton_density_scaling(fingerprint, PD=PD)
			signals[i,:] = self.add_noise(fingerprint)
		paras = np.zeros((nbnoise,6))
		# initial point for the NLLS for the parameters: m0s, T1f, T2f, R, T2s, PD
		p0 = [0.5,1,0.1,100,5e-2,0.5]
		for i in range(nbnoise):
			optpara, _ = sc.optimize.curve_fit(simu,self.x,signals[i,:],p0=p0,bounds=bounds)
			paras[i,:] = np.array(optpara)
		if save_name != '-1':
			np.save(os.path.join(path, 'paramsnlls'+save_name+'.npy'),paras)
		return np.std(paras, axis=0), np.mean(paras, axis=0)
