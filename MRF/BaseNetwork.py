import torch
import pandas as pd
import time
import copy
import importlib
import numpy as np
import os
from MRF.simulate_signal import simulation_with_grads


class BaseNetwork():
	"""
	Mother of Network classes that would contains the method for training.
	"""
	def __init__(self, name_model, loss, training_parameters, data_class, projection=None):
		self.data_class = data_class
		self.name_model = name_model
		self.loss       = loss
		self.trparas    = training_parameters
		self.projection = projection
		USE_CUDA = True and torch.cuda.is_available();
		self.device = torch.device('cuda' if USE_CUDA else 'cpu')
		
	def dico_save(self):
		""" Save the parameters of the instance of the class BaseNetwork."""
		dic = {}
		if self.projection is not None:
			dic = self.projection.dico_save()
		dic.update({'name_model': self.name_model, 'loss': self.loss})
		dic.update(self.data_class.dico_save())
		return dic

	def base_loss_function(self, outputs, params, para, size, CRB=None):
		"""
		Compute the loss function. 'outputs' is the output given by of the neural network and 'params' is the ground truth.
		"""
		if self.loss[para] == 'MSE-CRB':
			return torch.div((outputs.float() - params.float()).pow(2), CRB.float()).sum() / size
		elif self.loss[para] == 'MSE-Rela':  # MSE/(param+0.05)
			return torch.div((outputs.float() - params.float()).pow(2), params.float() + 0.05).sum() / size
		elif self.loss[para] == 'MSE-CRB-Scale':  # MSE/CRB*(1-param)
			return torch.mul(torch.div((outputs.float() - params.float()).pow(2), CRB.float()),
							 1.0 - params.float()).sum() / size
		elif self.loss[para] == 'MAE-Rela': #MAE/(param+0.001)
			#return torch.abs((outputs.float() - params.float()), params.float() + 0.05).sum() / size
			return torch.div((outputs.float() - params.float()).abs(), params.float()).sum() / size
		else:
			w = outputs.clone()
			w = w.detach()
			w = w.cpu()
			if np.isnan(w).any():
				print('nan exist in outputs in para:'+str(para))
				exit()
			return (outputs - self.transform_inv(params, para)).pow(2).sum() / size

	def loss_function(self, outputs, params, size, CRBs=None):
		"""
		Compute the loss function. 'outputs' is the output given by of the neural network and 'params' is the ground truth. 
		"""
		if len(self.trparas.params)==1:
			para = self.trparas.params[0]
			CRB = CRBs[:,0].reshape(-1) if self.data_class.CRBrequired else None
			# CRB = CRBs[:,para].reshape(-1) if self.data_class.CRBrequired else None
			return  self.base_loss_function(outputs.reshape(-1), params[:,para].reshape(-1), para, size, CRB=CRB)
		else:
			loss = 0
			for ind, para in enumerate(self.trparas.params):
				if self.data_class.CRBrequired and not CRBs is None:
					CRB = CRBs[:, ind]
				else:
					CRB = None
				# CRB = CRBs[:,para] if self.data_class.CRBrequired else None
				loss += self.base_loss_function(outputs[:,ind], params[:,para], para, size, CRB=CRB)
			return loss
			
	def compute_relative_errors(self, estimations, parameters, size):
		"""
		Compute the relative errors of the different parameters. 'estimations' is the outputs of the neural network and 'parameters' are the ground truth parameters.
		"""
		if len(self.trparas.params)==1:
			para = self.trparas.params[0]
			return  ( torch.abs( self.transform(estimations.reshape(-1), para)-parameters[:,self.trparas.params].reshape(-1)) /(parameters[:,self.trparas.params].reshape(-1)) ).sum(dim=0) / size
		else:
			error = torch.zeros(len(self.trparas.params))
			for ind, para in enumerate(self.trparas.params):
				error[ind] = ( torch.abs( self.transform(estimations[:,ind], para)-parameters[:,para]) /(parameters[:,para]) ).sum(dim=0) / size
			return error 
			
	def compute_absolute_errors(self, estimations, parameters, size):
		"""
		Compute the relative errors of the different parameters. 'estimations' is the outputs of the neural network and 'parameters' are the ground truth parameters.
		"""
		if len(self.trparas.params)==1:
			para = self.trparas.params[0]
			return  ( torch.abs( self.transform(estimations.reshape(-1), para)-parameters[:,self.trparas.params].reshape(-1))  ).pow(2).sum(dim=0) / size
		else:
			error = torch.zeros(len(self.trparas.params))
			for ind, para in enumerate(self.trparas.params):
				error[ind] = ( torch.abs( self.transform(estimations[:,ind], para)-parameters[:,para])  ).pow(2).sum(dim=0) / size
			return error 

	def compute_absolute_errors_over_CRBs(self, estimations, parameters, size, CRBs):
		"""
		Compute the relative errors of the different parameters. 'estimations' is the outputs of the neural network and 'parameters' are the ground truth parameters.
		"""
		if len(self.trparas.params)==1:
			para = self.trparas.params[0]
			CRB = CRBs.reshape(-1)
			return ( ( torch.abs( self.transform(estimations.reshape(-1), para)-parameters[:,self.trparas.params].reshape(-1))  ).pow(2) / CRB ).sum(dim=0) / size
		else:
			error = torch.zeros(len(self.trparas.params))
			for ind, para in enumerate(self.trparas.params):
				CRB = CRBs[:,ind]
				error[ind] = ( ( torch.abs( self.transform(estimations[:,ind], para)-parameters[:,para])  ).pow(2) / CRB  ).sum(dim=0) / size
			return error 
			

	def transform(self, outputs, para):
		"""
		Go from the output of the network to the estimated parameters.
		"""
		if self.loss[para] == 'MSE-Log':
			return (10**outputs)
		elif self.loss[para] == 'MSE-Inverse':
			return (1./outputs)
		elif self.loss[para] == 'MSE-Scaling':
			return rescale(outputs)
		else:
			return outputs
			
	def transform_inv(self, params, para):
		"""
		Go from the parameters to the target values defined by the loss type chosen.
		"""
		if self.loss[para] == 'MSE-Log':
			# print('MSE-log')
			return (torch.log10(params))
		elif self.loss[para] == 'MSE':
			# print('MSE')

			return params
		elif self.loss[para] == 'MSE-Inverse':
			return (1./params)
		elif self.loss[para] == 'MSE-Scaling':
			return scale(params)

	def study_estimator(self, network, signals):
		"""
		Return the mean and the standard deviation of the estimated parameters given from the neural network 'net' using the fingerprints given by 'signals'. 
		'signals' is built concatening noisy versions obtained from a same fingerprint.
		"""
		network.eval()
		with torch.no_grad():
			out = self.eval_net(network, signals)
			output = out.numpy()
			mean = np.mean(output, axis=0)
			std = np.std(output, axis=0)
		return std, mean
	  
	def eval_net(self, network, signals):
		"""
		Return the estimated parameters using the network 'netw' on the input batch of fingerprints 'signals'.
		"""
		outputs = network(signals)
		for ind, para in enumerate(self.trparas.params):
			outputs[:,ind] = self.transform(outputs[:,ind], para)
		return outputs
	
	def local_study(self, network, path):
		"""
		Method used in the interactive tool to compute the mean and the std of the estimator given by the Non Linear Least Squares and the neural network.
		It also computes the Cramer Rao Bound associated with the fingerprint.
		
		--> In order to use this method, you need to have already launched the script 'compute_nlls.py' located at the root of the project. In this script, we can define the set of 
		fingerprints that you want to study.
		"""
		nllsfiles = [filename for filename in os.listdir(path) if filename.startswith("paramsnlls")]
		parameters = np.load(os.path.join(path, 'true_parameters.npy'))
		PDs = np.load(os.path.join(path, 'true_PD.npy'))
		nb_points = len(PDs)

		
		STDnlls  = np.zeros((nb_points, 6))
		MEANnlls = np.zeros((nb_points, 6))
		STDnet   = np.zeros((nb_points, len(self.trparas.params)))
		MEANnet  = np.zeros((nb_points, len(self.trparas.params)))
		CRBs     = np.zeros((nb_points, len(self.trparas.params)))
		if self.projection.initialization == "Fixlayer":
			signals = self.projection.initialization_first_layer(network, 'cpu')
		for i in range(nb_points):
			m0s, T1, T2, r, T2s = parameters[i,0], parameters[i,1], parameters[i,2], parameters[i,3], parameters[i,4]
			PD = PDs[i]
			y,_ = simulation_with_grads.simulate_MT_ODE_with_grads(self.data_class.x, self.data_class.TR, self.data_class.t, m0s, T1, T2, r ,T1, T2s)
			fingerprint = y[:,0]
		
			# Non Linear Least Squares	
			nlls = np.load(os.path.join(path, 'paramsnlls'+str(i)+'.npy'))
			STDnlls[i,:] = np.std(nlls, axis=0)
			MEANnlls[i,:] = np.mean(nlls, axis=0)
			
			nbnoises = nlls.shape[0]
			
			# Network
			signals = np.zeros((nbnoises, len(fingerprint)))
			for j in range(nbnoises):
				signals[j,:], PD = self.data_class.proton_density_scaling(fingerprint, PD=PD)
				signals[j,:] = self.data_class.add_noise(signals[j,:])
			signals = torch.tensor(signals, dtype=torch.float, device='cpu')
			if self.projection is not None:
				signals = self.projection.project(signals)
			stdnet, meannet = self.study_estimator(network, signals)
			STDnet[i,:] = stdnet.reshape(-1)
			MEANnet[i,:] = meannet.reshape(-1)
			
			# Cramer Rao Bound
			CRBs[i,:] = self.data_class.compute_CRBs(y, PD, nbnoises=nbnoises)
		return MEANnlls, STDnlls, MEANnet, STDnet, CRBs, parameters