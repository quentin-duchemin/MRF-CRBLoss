import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import data
from ..BaseModel import * 


class model(BaseModel):
	def __init__(self, nb_params=None, projection=None, ghost=False):
		super(model, self).__init__(True, True, nb_params=nb_params, projection=projection, ghost=ghost)
		if not self.ghost:
			self.assert_projection_defined()
			self.assert_joint_learning()
			if self.projection.complex:
				if self.projection.initialization != 'Fixlayer':
					self.fc1_real = nn.Linear(1142*2, self.projection.dimension_projection,bias=False)
					self.fc1_imag = nn.Linear(1142*2, self.projection.dimension_projection, bias=False)
				self.fc20 = nn.Linear(2*self.projection.dimension_projection, 128)
			else:
				self.fc1 = nn.Linear(1142, self.projection.dimension_projection)
				self.fc20 = nn.Linear(self.projection.dimension_projection, 128)
			self.bn20 = nn.BatchNorm1d(128)
			self.fc2 = nn.Linear(128, 256)
			self.bn2 = nn.BatchNorm1d(256)
			self.fc3 = nn.Linear(256, 512)
			self.bn3 = nn.BatchNorm1d(512)
			self.fc4 = nn.Linear(512, 1024)
			self.bn4 = nn.BatchNorm1d(1024)
			self.fc5 = nn.Linear(1024, 768)
			self.bn5 = nn.BatchNorm1d(768)
			self.fc6 = nn.Linear(768, 640)
			self.bn6 = nn.BatchNorm1d(640)
			self.fc7 = nn.Linear(640, 512)
			self.bn7 = nn.BatchNorm1d(512)
			self.fc8 = nn.Linear(512, 384)
			self.bn8 = nn.BatchNorm1d(384)
			self.fc9 = nn.Linear(384, 256)
			self.bn9 = nn.BatchNorm1d(256)
			self.fc10 = nn.Linear(256, 192)
			self.bn10 = nn.BatchNorm1d(192)
			self.fc11 = nn.Linear(192, 128)
			self.bn11 = nn.BatchNorm1d(128)
			self.fc12 = nn.Linear(128, 2*self.projection.dimension_projection)
			self.bn12 = nn.BatchNorm1d(2*self.projection.dimension_projection)
			self.fc13 = nn.Linear(2*self.projection.dimension_projection, 10)
			self.bn13  = nn.BatchNorm1d(10)
			self.fc14 = nn.Linear(10, 6)
			self.bn14 = nn.BatchNorm1d(6)
			self.fc15 = nn.Linear(6, self.nb_params)

	def forward(self, s):
		batch = s.size()[0]
		if self.projection.initialization != 'Fixlayer': #in training, always set it to random
			if self.projection.complex:
				s_real = self.fc1_real(s[:, :, 0])  # (batch, 1142)
				s_imag = self.fc1_imag(s[:, :, 1])  # (batch, 1142)
				proj = torch.cat((s_real, s_imag), 1)  # (batch, 18)
			else:
				proj = self.fc1(s)
		else:
			proj = s # in invivo test, always set it to Fixlayer

		if self.projection.normalization == "After_projection":
			proj = self.normalization_post_projection_complex(proj)
			
		s128 = self.fc20(proj)
		s256= self.fc2(F.relu(self.bn20(s128)))
		s512 = self.fc3(F.relu(self.bn2(s256)))
		s = F.relu(self.bn4(self.fc4(F.relu(self.bn3(s512)))))
		s = F.relu(self.bn5(self.fc5(s)))
		s = F.relu(self.bn6(self.fc6(s)))
		s = self.fc7(s)
		s = F.relu(self.bn8(self.fc8(F.relu(self.bn7(s + s512)))))
		s = self.fc9(s)
		s = F.relu(self.bn10(self.fc10(F.relu(self.bn9(s+s256)))))
		s = self.fc11(s)
		s = self.fc12(F.relu(self.bn11(s+s128)))
		s = F.relu(self.bn13(self.fc13(F.relu(self.bn12(s+proj)))))
		s = F.relu(self.bn14(self.fc14(s)))
		s = self.fc15(s)
		return s