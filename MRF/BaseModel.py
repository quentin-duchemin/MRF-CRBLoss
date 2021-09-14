import numpy as np
import torch
import os
from torch.nn import Module
import torch.nn.functional as F
from torch.utils import data



class BaseModel(Module):
	"""
	Each new network should inherit from this class and be saved in the folder 'models'.
	"""
	def __init__(self, joint, start_by_projection, nb_params=None, projection=None, ghost=False):
		super(BaseModel, self).__init__()
		self.start_by_projection  = start_by_projection
		self.projection           = projection
		self.nb_params 			  = nb_params
		self.joint                = joint
		self.ghost                = ghost 
		
	def assert_projection_defined(self):
		"""
		Ensure that 'projection' is defined if the network can allow to start with a projection of the input onto a low dimensional subspace.
		"""
		assert((self.start_by_projection and self.projection.start_by_projection), "Using a projection requires to define the attributes normalization and dimension_projection of the class BaseModel")

	def assert_joint_learning(self):
		"""
		Ensure that 'nb_params' is defined if the network learns jointly the parameters.
		"""
		assert((self.joint and self.nb_params is not None), "Learning jointly the parameters requires to define the attribute nb_params of the class BaseModel")

	def normalization_post_projection(self, proj):
		"""
		Perform normalization after the projection of the fingerprints onto a low dimensional subspace.
		"""
		proj = proj / torch.norm(proj,dim=1).unsqueeze(1).repeat(1, self.projection.dimension_projection)
		return proj

	def normalization_post_projection_complex(self, proj):
		"""
		Perform normalization after the projection of the fingerprints onto a low dimensional subspace.
		"""
		#proj = proj / torch.norm(proj,dim=1).unsqueeze(1).repeat(1, 2*self.projection.dimension_projection)
		proj = proj / torch.norm(proj, dim=1).unsqueeze(1).repeat(1, proj.shape[1])
		return proj

	def forward(self, signals):
		"""
		Define the architecture of the network.
		"""
		raise NotImplementedError("This method forward has to be implemented in the class inheriting from BaseModel.")
