import torch
import numpy as np



class Performances():
	"""
	Class designed to handle the computations and the definition of the validation loss and errors.
	"""
	def __init__(self, validation_settings):
		self.losses                              = []
		self.training_relative_errors            = []
		self.training_absolute_errors            = []
		self.training_absolute_errors_over_CRBs  = []
		for key, value in validation_settings.items():
			setattr(self, key, value)
		self.gradients                           = []
			
	def dico_save(self):
		""" Save the parameters of the instance of the class Performances."""
		return (self.__dict__).copy()

	
	def loss_function(self,outputs,params,size):
 		raise NotImplementedError("Must override loss_function.")
        
	def compute_relative_errors(self, estimations_validation, parameters, size):
 		raise NotImplementedError("Must override compute_relative_errors.")
		
	def validation_step(self, estimations_validation):
		"""
		Compute the loss and the relative errors on the parameters on the validation dataset. The parameter 'estimation_validation' represents the estimation of the network for the parameters on the validation dataset.
		"""
		self.losses_validation.append((self.loss_function(estimations_validation,self.params_validation, self.validation_size*len(self.trparas.params), CRBs = self.CRBs_validation)).cpu().detach().numpy())
		self.validation_relative_errors.append((self.compute_relative_errors(estimations_validation, self.params_validation, self.validation_size)).cpu().detach().numpy())
		CRBs = self.CRBs_validation[:self.small_validation_size,:] if self.data_class.CRBrequired else None
		self.losses_small_validation.append((self.loss_function(estimations_validation[:self.small_validation_size],self.params_validation[:self.small_validation_size], self.small_validation_size*len(self.trparas.params), CRBs = CRBs)).cpu().detach().numpy())
		self.small_validation_relative_errors.append((self.compute_relative_errors(estimations_validation[:self.small_validation_size], self.params_validation[:self.small_validation_size], self.small_validation_size)).cpu().detach().numpy())
		self.validation_absolute_errors.append((self.compute_absolute_errors(estimations_validation, self.params_validation, self.validation_size)).cpu().detach().numpy())
		if self.data_class.CRBrequired:
			self.validation_absolute_errors_over_CRBs.append((self.compute_absolute_errors_over_CRBs(estimations_validation, self.params_validation, self.validation_size, self.CRBs_validation)).cpu().detach().numpy())

	def init_validation(self, iscomplex):
		"""
		Define the validation dataset.
		"""
		if iscomplex:
			if self.validation:
				self.losses_validation, self.validation_relative_errors, self.losses_small_validation, self.small_validation_relative_errors = [], [], [], []
				self.validation_absolute_errors, self.validation_absolute_errors_over_CRBs = [], []

				self.CRBs_validation = None
				dico_validation, params_validation, CRBs_validation = self.data_class.load_data(1)
				num_files = 1
				ndata = params_validation.shape[0]
				count = ndata
				while count < self.validation_size:
					num_files += 1
					inputs, parameters, CRBs = self.data_class.load_data(num_files)
					dico_validation = np.concatenate((dico_validation, inputs), axis=0)
					params_validation = np.concatenate((params_validation, parameters), axis=0)

					if self.data_class.CRBrequired:
						CRBs_validation = np.concatenate((CRBs_validation, CRBs), axis=0)
						CRBs_validation = CRBs_validation[:self.validation_size, :]
					count += ndata
				self.num_files_validation = num_files
				dico_validation = dico_validation[:self.validation_size, :]
				params_validation = params_validation[:self.validation_size, :]
				dico_validation, PD = self.data_class.proton_density_scaling_B0_complex(dico_validation)

				dico_validation = torch.tensor(self.data_class.add_noise_batch_B0(dico_validation), dtype=torch.float)
				self.dico_validation = (dico_validation).to(device=self.device)

				self.params_validation = torch.tensor(params_validation, dtype=torch.float, device='cpu')
				self.PD_validation = PD
				if self.data_class.CRBrequired:
					CRBs_validation = CRBs_validation[:self.validation_size, :]
					PD = PD.reshape(-1, 2)  # (b,2)

					PD_norm = PD[:, 0] ** 2 + PD[:, 1] ** 2
					PD_norm = PD_norm.reshape(-1, 1)

					CRBs_validation[:,:3] /=  np.tile(PD_norm,(1,3))
					CRBs_validation *= self.data_class.noise_level **2
					self.CRBs_validation = torch.tensor(CRBs_validation[:, self.trparas.params], dtype=torch.float,
														device='cpu')
			else:
				self.num_files_validation = 0

		else:
			if self.validation:
				self.losses_validation, self.validation_relative_errors, self.losses_small_validation, self.small_validation_relative_errors = [], [], [], []
				self.validation_absolute_errors, self.validation_absolute_errors_over_CRBs = [], []
				self.CRBs_validation = None
				# use data as validation starting from the 1st files
				dico_validation, params_validation, CRBs_validation = self.data_class.load_data(1)
				num_files = 1
				ndata = params_validation.shape[0]
				count = ndata
				while count < self.validation_size:
					num_files += 1
					inputs, parameters, CRBs = self.data_class.load_data(num_files)
					dico_validation = np.concatenate((dico_validation, inputs), axis=0)
					params_validation = np.concatenate((params_validation, parameters), axis=0)

					if self.data_class.CRBrequired:
						CRBs_validation = np.concatenate((CRBs_validation, CRBs), axis=0)
						CRBs_validation = CRBs_validation[:self.validation_size, :]
					count += ndata
				self.num_files_validation = num_files
				dico_validation = dico_validation[:self.validation_size, :]
				params_validation = params_validation[:self.validation_size, :]
				dico_validation, PD = self.data_class.proton_density_scaling(dico_validation)
				dico_validation = torch.tensor(self.data_class.add_noise_batch(dico_validation), dtype=torch.float)
				self.dico_validation = (dico_validation).to(device=self.device)
				self.params_validation = torch.tensor(params_validation, dtype=torch.float, device='cpu')
				self.PD_validation = PD
				if self.data_class.CRBrequired:
					CRBs_validation = CRBs_validation[:self.validation_size, :]
					PD = PD.reshape(-1,1)
					CRBs_validation[:,:3] /=  np.tile(PD**2,(1,3))
					CRBs_validation *= self.data_class.noise_level **2
					self.CRBs_validation = torch.tensor(CRBs_validation[:, self.trparas.params], dtype=torch.float,
														device='cpu')
			else:
				self.num_files_validation = 0

