from enum import Enum
import numpy as np

#TR = 4.5e-3
TR = 3.5e-3
params_min_values = np.array([0.001, 0.01, 0.01,  10,  1e-3])
params_max_values = np.array([0.5,   5,    3,    100, 1e-1])

nametoparam = {'The three parameters':[0,1,2], 'The3_b1':[0,1,2,7], 'The3_b0_b1':[0,1,2,6,7], 'b0':[6],'b1':[7],'m0s':[0],'T1':[1],'T2':[2],'T1 and T2':[1,2], 'R':[3], 'T2s':[4], 'The four parameters':[0,1,2,3], 'The five parameters':[0,1,2,3,4]}
paramtoname =  {0:'m0s', 1:'T1', 2:'T2', 3:'R', 4:'T2s', 5:'M0', 6:'b0', 7:'b1'}
paramtolatexname =  {0:'$m_0^s$', 1:'$T_1^f$', 2:'$T_2^f$', 3:'$R$', 4:'$T_2^s$', 5:'$M_0$', 6:'$B_0$', 7:'$B_1$'}


class Training_parameters():
	'''
	Define the hyperparameters to train the neural network.
	'''
	def __init__(self, batch_size, nb_iterations, nb_epochs, params, normalization, isdatacomplex):
		self.batch_size    = batch_size
		self.nb_iterations = nb_iterations
		self.nb_epochs     = nb_epochs
		self.params        = params
		self.normalization = normalization
		self.complex       = isdatacomplex
		
	def dico_save(self):
		return self.__dict__	

class Enumerate(Enum):
	@classmethod
	def list(self):
		return [e.value for e_name, e in self.__members__.items()]
			
class Loss(Enumerate):
	'''
	Define the type of loss function used to train the neural network ON A SPECIFIC parameter.
	This means that you can combine different types of loss if you estimate several parameters.
	'''
	#: The loss function is defined as the mse on the parameter where we divide each squared error by the Cramer Rao Bound. 
	MSECRB     = 'MSE-CRB'
	#: The loss function is defined as the mse considering the ground truth as the log of the parameter.
	MSELOG     = "MSE-Log"
	#: The loss function is defined as the mse.
	MSE        = "MSE"
	#: The loss function is defined as the mse considering the ground truth as the inverse of the parameter.
	MSEInv     = "MSE-Inverse"
	#: The loss function is defined as the mse considering the ground truth as scaled parameter (i.e. we perform a scaling on the parameter so that the scaled value can vary only between 0 and 1).
	MSEScaling = "MSE-Scaling"
	# relative difference, SCQ method
	MAERela = 'MAE-Rela'
	
	@classmethod
	def CRBrequired(self, loss):
		'''Put the name of the loss in the list if it requires to compute the CRB wrt the parameters.'''
		if loss in ['MSE-CRB']:
			return True
		else:
			return False
	
class Normalization(Enumerate):
	'''
	Define the way you want to normalize (or not) the input signal.
	'''
	#: Option to avoid normalization.
	WHITHOUT   = "Without"
	#: Option to normalize the fingerprints just after adding the noise realization, i.e. the inputs of the neural network are the normalized fingerprints.
	NOISYINPUT = "Noisy_input"
	#: Option to normalize the output of the first linear layer supposed to perform the projection onto a lower dimensional subspace. It is available only if your network starts with a projection.
	AFTERPROJ  = "After_projection"
	
class Optimizer(Enumerate):
	'''
	Select the optimizer used for the training.
	'''
	#: Option to use a stochastic gradient descent to perform the backpropagation with a momentum of 0.9.
	SGD = 'SGD'
	#: Option to use the algorithm Adam to perform the backpropagation.
	ADAM = 'Adam'

class NoiseType(Enumerate):
	'''
	Describe the way you want to define the noise.
	'''
	#: Option to define the noise on the inputs considering the SNR defined as the ratio between the L2 norm of the signal and L2 norm of the noise. The value of the SNR should be specified with the attribute `noise_level`.   
	SNR      = "SNR"
	#: Option to define the noise on the inputs adding a centered gaussian noise with standard deviation given by the attribute `noise_level`.   
	STANDARD = "Standard"

class Initialization(Enumerate):
	'''
	Specify the way to deal with the first linear layer if your network starts with a projection of the signal onto a low dimensional subspace.
	'''
	#: Allow to initialize the first linear layer performing the projection onto a low dimensional subspace using the basis functions given by an online PCA algorithm. The file containing this basis needs t be saved in the folder 'online_PCA' and its name is given by the attribute `namepca`.
	PCA      = "PCA"
	#: Allow to initialize randomly the first linear layer performing the projection onto a low dimensional subspace.
	RANDOM   = "Random"
	#: Allow to use a fixed difinition for the first linear layer performing the projection onto a low dimensional subspace. Typically, we use the basis functions given by an online PCA algorithm. The file containing this basis needs t be saved in the folder 'online_PCA' and its name is given by the attribute `namepca`.
	FIXLAYER = "Fixlayer"
	
class Sampling(Enumerate):
	'''
	Define the way you want to sample the parameters. It is related with the `sample` method of the class `Data_class`.
	'''
	#: The log uniform sampling is used to sample T1, T2 and T2f. The other parameters are sampled using a uniform sampling. 
	LOG     = "Log"
	#: All the parameters are drawn using a uniform sampling: You can define the min and max values of the different parameters using the attributes `min_values` and `max_values` in the Online framework.
	YOUNIFORM = "YOUniform"
	#: All the parameters are drawn using a uniform sampling with a specific and fixed strategy.
	Uniform = "Uniform"
	#: All the parameters are drawn using a gaussian sampling with a particular mean and vairance with a constraint and their minimum value.
	GAUSSIAN = "Gaussian"