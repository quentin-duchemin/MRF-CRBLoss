Developer
*********

> We explain here the way you can add new features to the model.

## A new sampling strategy

- In an offline framework, you do not need to modify anything to project. You just need to built the files contiang the fingerprints using the sampling strategy that you want ot use. Once you have saved a .txt file containing the urls of the files with the fingerprints, we can load them locally in this folder. Then you can launch your job following the method described [here](https://master_thesis.github.io/docs/build/html/runcode.html).

- In an Online framework:

-- *First*, open the file *Training_parameters.py* located in the folder MRF and add in the enumerate class *Sampling* the name of the sampling strategy that you want to define. 

-- *Then*, you only need to modify the method *sample* of the class *MRF.Online.Data_class* by adding an **elif self.sampling == the-name-of-your-sampling-strategy** followed by the sampling of the parameters. You can take example on the sampling strategies already implemented to correctly write your own sampling strategy.

----

## A new loss

-- *First*, open the file *Training_parameters.py* located in the folder MRF and add in the enumerate class *Loss* the name of the Loss that you want to define. If in the definition of the loss you need to have the knowledge of the Cramer Rao Bound, you have to modify the classmethod *CRBrequired* of the class Loss adding to the list the name given to your loss. For example, let's suppose that the name of your new loss is *"newloss"* and that it needs the Cramer Rao Bounds to be computed, then the classmethod *CRBrequired* should be :

``` python
	@classmethod
	def CRBrequired(self, loss):
		if loss in ['MSE-CRB', 'newloss']:
			return True
		else:
			return False
```

-- *Then*, you only need to modify at most **three** methods: *transform*, *transform_inv* and *base_loss_function* of the class *MRF.BaseNetwork* by adding an **elif self.loss == the-name-of-your-loss** followed by transformation on the parameters that you want to perform and the inverse of it. You can take example on the transformation already implemented to correctly complete the code.

----

## A new type of noise

-- *First*, open the file *Training_parameters.py* located in the folder MRF and add in the enumerate class *NoiseType* the name of the type of noise that you want to define. 

-- *Then*, you only need to modify the **two** methods *add_noise* and *add_noise_batch* of the class *MRF.BaseData_class* by adding an **elif self.noise_type == the-name-of-your-noise** followed by noise that you want to add to the fingerprints. You can take example on the transformation already implemented to correctly complete the code. 

-- This new noise realization also requires to define correctely the Cramer Rao Bound for the parameters. Indeed, a new noise realization will deeply influence the definition. Thus, you also will have to define the Cramer Rao Bound in the method `compute_CRBs` of the class BaseNetwork if you still want to be able to use the option *NN VS NLLS and CRB* in the interactive tool deisgned to visualize your results.

**Advice**: Don't hesitate to use the attribute *noise_level* to define your noise according to this mutable parameter. You can define it the way that suits you.

----



## A new optimizer

- *First*, in the module **Training_parameters**, you have to add the name given to this new optimizer in the class `Optimizer`.
- *Then*, open the files *Network.py* (located in the folder **Offline** and **Online**) and add an 'elif nameoptimizer == {the name of your new optimizer}' in the method *train* of the class `Network`.

----

## A new architecture

In order to define a new architecture, you need to create a class named `model` which will inherit from the class `BaseModel`. The python file will have to be saved in the folder MRF/models. The code below give you the way you should write correctly the python file.

```python
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import data
from ..BaseModel import * 


class model(BaseModel):
	def __init__(self, nb_params=None, projection=None, ghost=False):

		# BOOL1 : boolean which should be True if your network outputs all the parameters jointly (i.e. the last layer returns a vector of size equals to the number of parameters learned)
		# BOOL1 : boolean which should be False if your network outputs separately the parameters (e.g. the only layer shared by the different parameters is the first layer and then different architecture are designed for each parameter)

		# BOOL2 : boolean which should be True if your network starts with a projection and False otherwise

		super(model, self).__init__(BOOL1, BOOL2, nb_params=nb_params, projection=projection, ghost=ghost)
		if not self.ghost:
			# if your network starts with a projection, you should name this layer *fc1*
			self.fc1 = nn.Linear(666, self.projection.dimension_projection)

	def forward(self, s):
		# if your network start with a projection, you should allow to normalize the projected signal and to use a fixlayer
		if self.projection.initialization != 'Fixlayer':
			s = self.fc1(s)
		if self.projection.normalization:
			s = self.normalization_post_projection(s)

		...

		return s

```

**Advice**: Don't hesitate to look the different architectures already implemented in the package `MRF.models` if you need further help to write your own network.
