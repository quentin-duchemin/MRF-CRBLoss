import torch
import sys
import pandas as pd
import time
import copy
import importlib
import scipy as sc
import os


class Projection():
	"""
	Class gathering the parameters to characterize the first linear layer which performs the projection of the input signal onto a low dimensional subspace (if the chosen architecture includes a projection).
	"""
	def __init__(self, start_by_projection, dimension_projection, initialization, normalization, namepca):
		print('namepca')
		print(namepca)
		self.start_by_projection  = start_by_projection
		self.initialization       = initialization
		self.normalization        = normalization
		self.dimension_projection = dimension_projection
		self.namepca              = namepca
		# my_path                   = os.path.abspath(os.path.dirname(__file__))
		# path                      = os.path.join(my_path, '../online_PCA/'+namepca)
		# self.eigenvectors         = sc.io.loadmat(path)['basis']
		self.eigenvectors = []#Ssc.io.loadmat(namepca)['u']

	def initialization_first_layer(self, net, device):
		"""
		Method allowing to initialize by yourself the weight of the first linear layer. You can for example use the basis functions given by an online PCA algorithm.
		"""
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
		if self.initialization == 'Fixlayer':
			inputs = inputs.mm(self.eigenvectors)
		return inputs

	def project_complex(self, inputs):
		"""
		If a fixed first linear layer is used, this method performs the projection of the input signal onto the low dimensional subspace.
		"""
		if self.initialization == 'Fixlayer':
			s_real = inputs[:, :, 0]  # (batch, 1142)
			s_imag = inputs[:, :, 1]  # (batch, 1142)

			# for the multi-coef model
			# s_real = s_real.reshape(batch, -1) #(batch, 3408)
			# s_imag = s_imag.reshape(batch, -1) #(batch, 3408)

			# s_comb = torch.cat((s_real, s_imag), 1) #(batch, 3408*2)
			proj_real = self.fc1_real(s_real) - self.fc1_imag(s_imag)  # (batch, 9)
			proj_imag = self.fc1_real(s_imag) + self.fc1_imag(s_real)  # (batch, 9)

			proj = torch.cat((proj_real, proj_imag), 1)  # (batch, 18)



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