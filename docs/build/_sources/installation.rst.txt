Installation
------------

Prerequisites
~~~~~~~~~~~~~~

MRF-CRBLoss is a Python package allowing to estimate biophysical parameters from MRF scans. In the following, we describe the steps to follow to get a correct environment to use our package.

conda prerequisites
###################

1. Install Conda. We typically use the Miniconda_ Python distribution. Use Python version >=3.7.

2. Create a new conda environment::

    conda create -n mrf-env python=3.7

3. Activate your environment::

    source activate mrf-env

python prerequisites
####################

1. Install Python_, we prefer the `pyenv <https://github.com/pyenv/pyenv/>`_ version management system, along with `pyenv-virtualenv <https://github.com/pyenv/pyenv-virtualenv/>`_.

2. Install PyTorch_. If you have an Nvidia GPU, be sure to install a version of PyTorch that supports it.

.. _Miniconda: https://conda.io/miniconda.html
.. _Python: https://www.python.org/downloads/
.. _PyTorch: http://pytorch.org


Downloading the package
~~~~~~~~~~~~~~~~~~~~~~~

1. clone the repository::
	
	git clone https://github.com/quentin-duchemin/MRF-CRBLoss.git


2. install the required python packages in the virtualenv::

	pip install -r requirements_MRF.txt