Developer
=========

You are interested by our project and you want to use our code for your own research projects? This is the place to be ! 

In that case, you may require to modify the code to fit your specific expectations. In this section, we provide additional information on our implementation that should allow you to easily modify the code for your needs (and in particular to add new functionalities).



A new architecture
------------------

In order to define a new architecture, you need to create a class named `model` which will inherit from the class `BaseModel`. The python file will have to be saved in the folder MRF/models. The code below give you the way you should write correctly the python file.

.. code-block:: python

	import numpy as np
	import torch
	import torch.nn as nn
	import torch.nn.functional as F
	from torch.utils import data
	from ..BaseModel import * 


	class model(BaseModel):
		def __init__(self, nb_params=None, projection=None, ghost=False):


			"""
			Parameters
			----------
			nb_params : int
			Number of parameters that we want to estimate
			projection : Projection
			An instance of the class `Projection` that contains all the details regarding the way we perform the projection of the input signal.
			ghost : boolean
			When set to `True`, this can allow us to simply know if the model estimate jointly each parameters and if it starts with a projection layer (see the booleans `Bool1` and `Bool2`below).
			"""

			super(model, self).__init__(BOOL1, BOOL2, nb_params=nb_params, projection=projection, ghost=ghost)

			# Bool1 is a boolean which should be `True` if your network outputs all the parameters jointly (i.e. the last layer returns a vector of size equals to the number of parameters learned). On the contrary, Bool1 which should be `False` if your network outputs separately the parameters (e.g. the only layer shared by the different parameters is the first layer and then different architecture are designed for each parameter).

			# Bool2 is a boolean which should be `True` if your network starts with a projection and False otherwise

			if not self.ghost:
				if self.projection.complex:
					# if the input signal are complex valued
					if self.projection.initialization != 'Fixlayer':
						# if your network starts with a projection, you should name these layers *fc1_real* and *fc1_imag*
						self.fc1_real = nn.Linear(1142*2, self.projection.dimension_projection,bias=False)
						self.fc1_imag = nn.Linear(1142*2, self.projection.dimension_projection, bias=False)
						self.fc2 = nn.Linear(2*self.projection.dimension_projection, 128)
				else:
					# if your network starts with a projection, you should name this layer *fc1*
					self.fc1 = nn.Linear(1142, self.projection.dimension_projection)
					self.fc2 = nn.Linear(self.projection.dimension_projection, 128)
				
				# Please note that in the above code, `1142` corresponds to the length of the input signals and you might want to change it to fit your case.

				self.fc3 = nn.Linear(128, self.nb_params)



		def forward(self, s):
			batch = s.size()[0]
			# if your network starts with a projection, you should allow to normalize the projected signal and to use a fixlayer
			if self.projection.initialization != 'Fixlayer':
				if self.projection.complex:
					s_real = self.fc1_real(s[:, :, 0])  # (batch, 1142)
					s_imag = self.fc1_imag(s[:, :, 1])  # (batch, 1142)
					proj = torch.cat((s_real, s_imag), 1)
				else:
					proj = self.fc1(s)
			else:
				proj = s

			if self.projection.normalization == "After_projection":
				proj = self.normalization_post_projection_complex(proj)
				
			s = self.fc2(proj)
			s = self.fc3(s)

			return s

**Advice**: Don not hesitate to look the different architectures already implemented in the package `MRF.models` if you need further help to write your own network.

A new loss
----------

-- *First*, open the file *Training_parameters.py* located in the folder MRF and add in the enumerate class *Loss* the name of the Loss that you want to define. If in the definition of the loss you need to have the knowledge of the Cramer Rao Bound, you have to modify the classmethod *CRBrequired* of the class Loss adding to the list the name given to your loss. For example, let's suppose that the name of your new loss is *"newloss"* and that it needs the Cramer Rao Bounds to be computed, then the classmethod *CRBrequired* should be :

.. code-block:: python

	@classmethod
	def CRBrequired(self, loss):
		if loss in ['MSE-CRB', 'newloss']:
			return True
		else:
			return False


-- *Then*, you only need to modify at most **three** methods: *transform*, *transform_inv* and *base_loss_function* of the class *MRF.BaseNetwork* by adding an **elif self.loss == the-name-of-your-loss** followed by transformation on the parameters that you want to perform and the inverse of it. You can take example on the transformation already implemented to correctly complete the code.


A new type of noise
-------------------

-- *First*, open the file *Training_parameters.py* located in the folder MRF and add in the enumerate class *NoiseType* the name of the type of noise that you want to define. 

-- *Then*, you only need to modify the **two** methods *add_noise* and *add_noise_batch* of the class *MRF.BaseData_class* by adding an **elif self.noise_type == the-name-of-your-noise** followed by noise that you want to add to the fingerprints. You can take example on the transformation already implemented to correctly complete the code. 

-- This new noise realization also requires to define correctely the Cramer Rao Bound for the parameters. Indeed, a new noise realization will deeply influence the definition. Thus, you also will have to define the Cramer Rao Bound in the method `compute_CRBs` of the class BaseNetwork if you still want to be able to use the option *NN VS NLLS and CRB* in the interactive tool deisgned to visualize your results.

**Advice**: Don't hesitate to use the attribute *noise_level* to define your noise according to this mutable parameter. You can define it the way that suits you.

A new optimizer
---------------

- *First*, in the module **Training_parameters**, you have to add the name given to this new optimizer in the class `Optimizer`.

- *Then*, open the files *Network.py* (located in the folder **Offline** and **Online**) and add an 'elif nameoptimizer == {the name of your new optimizer}' in the method *train* of the class `Network`.


A new sampling strategy
-----------------------

You do not need to modify anything to project. You just need to built the files containing the fingerprints using the sampling strategy that you want ot use. Once you have saved a .txt file containing the urls of the files with the fingerprints, we can load them locally in this folder. Then you can launch your job following the method described `here <https://quentin-duchemin.github.io/MRF-CRBLoss/build/tutorials/notebooks/quickstart.html>`_.

