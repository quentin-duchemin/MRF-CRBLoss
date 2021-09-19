import torch
import sys
import pandas as pd
import time
import copy
import importlib
import scipy as sc
import os
import numpy as np


class Projection():
	"""
	Class gathering the parameters to characterize the first linear layer which performs the projection of the input signal onto a low dimensional subspace (if the chosen architecture includes a projection).
	"""
	def __init__(self, start_by_projection, dimension_projection, initialization, normalization, namepca, complex, ghost=False):
		self.start_by_projection  = start_by_projection
		self.initialization       = initialization
		self.normalization        = normalization
		self.dimension_projection = dimension_projection
		self.namepca              = namepca
		self.complex              = complex
		if not ghost:
			if self.initialization == 'Fixlayer' or self.initialization == 'PCA':
				if self.complex:
					self.eigenvectors = np.conj(sc.io.loadmat(namepca)['u'])
				else:
					self.eigenvectors = sc.io.loadmat(namepca)['u']

	def initialization_first_layer(self, net, device):
		"""
		Method allowing to initialize by yourself the weight of the first linear layer. You can for example use the basis functions given by an online PCA algorithm.
		"""
		if self.complex:
			if self.initialization == 'PCA':
				self.eigenvectors = self.eigenvectors[:,:self.dimension_projection]
				self.fc1_real_weight = self.eigenvectors.real.T
				self.fc1_imag_weight = self.eigenvectors.imag.T

				self.fc1_real_weight =  torch.tensor(self.fc1_real_weight, dtype=torch.float, device=device)
				self.fc1_imag_weight =  torch.tensor(self.fc1_imag_weight, dtype=torch.float, device=device)


				net.fc1_real.weight.data = self.fc1_real_weight
				net.fc1_imag.weight.data = self.fc1_imag_weight


			elif self.initialization == 'Fixlayer':
				self.eigenvectors = self.eigenvectors[:,:self.dimension_projection]
				self.fc1_real_weight = self.eigenvectors.real
				self.fc1_imag_weight = self.eigenvectors.imag

				self.fc1_real_weight =  torch.tensor(self.fc1_real_weight, dtype=torch.float, device=device)
				self.fc1_imag_weight =  torch.tensor(self.fc1_imag_weight, dtype=torch.float, device=device)

			return net
		else:
			if self.initialization == 'PCA':
				eigenvectors = torch.tensor(self.eigenvectors[:self.dimension_projection,:],dtype=torch.float,device=device)
				net.fc1.weight.data = eigenvectors
			elif self.initialization == 'Fixlayer':
				self.eigenvectors = torch.tensor(self.eigenvectors[:self.dimension_projection,:].T, dtype=torch.float, device=device)
			return net

	def initialization_first_layer_complex(self, net, device):
		"""
		Method allowing to initialize by yourself the weight of the first linear layer. You can for example use the basis functions given by an online PCA algorithm.
		"""
		if self.initialization == 'PCA':
			eigenvectors = torch.tensor(self.eigenvectors[:self.dimension_projection,:],dtype=torch.float,device=device)
			net.fc1.weight.data = eigenvectors
		elif self.initialization == 'Fixlayer':
			self.eigenvectors = torch.tensor(self.eigenvectors[:self.dimension_projection,:].T, dtype=torch.float, device=device)
		return net


	def project(self, inputs):
		"""
		If a fixed first linear layer is used, this method performs the projection of the input signal onto the low dimensional subspace.
		"""
		if self.complex:
			if self.initialization == 'Fixlayer':
				s_real = inputs[:, :, 0]  # (batch, 1142)
				s_imag = inputs[:, :, 1]  # (batch, 1142)
				proj_real = s_real.mm(self.fc1_real_weight) - s_imag.mm(self.fc1_imag_weight)  # (batch, 9)
				proj_imag = s_imag.mm(self.fc1_real_weight) + s_real.mm(self.fc1_imag_weight)    # (batch, 9)

				proj = torch.cat((proj_real, proj_imag), 1)  # (batch, 18)
				inputs = proj

			return inputs
		else:
			if self.initialization == 'Fixlayer':
				inputs = inputs.mm(self.eigenvectors)
			return inputs

	def dico_save(self):
		dic = {
				'start_by_projection'  : self.start_by_projection,
				'initialization'       : self.initialization,
 				'normalization'        : self.normalization,
 				'dimension_projection' : self.dimension_projection,
 				'namepca'              : self.namepca
            }
		return dic