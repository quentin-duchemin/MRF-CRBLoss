Installation
------------

Prerequisites
~~~~~~~~~~~~~~

scvi-tools can be installed via conda or pip. If you don't know which to choose, we recommend conda for beginner users.

conda prerequisites
###################

1. Install Conda. We typically use the Miniconda_ Python distribution. Use Python version >=3.7.

2. Create a new conda environment::

    conda create -n mrf-env python=3.7

3. Activate your environment::

    source activate mrf-env

pip prerequisites
#################

1. Install Python_, we prefer the `pyenv <https://github.com/pyenv/pyenv/>`_ version management system, along with `pyenv-virtualenv <https://github.com/pyenv/pyenv-virtualenv/>`_.

2. Install PyTorch_. If you have an Nvidia GPU, be sure to install a version of PyTorch that supports it.

.. _Miniconda: https://conda.io/miniconda.html
.. _Python: https://www.python.org/downloads/
.. _PyTorch: http://pytorch.org


Conda
~~~~~

::

    conda install mrf-crbloss -c bioconda -c conda-forge

Pip
~~~

::

    pip install mrf-crbloss

Through pip with packages to run notebooks. This installs scanpy, etc.::

    pip install mrf-crbloss[tutorials]

Nightly version - clone this repo and run::

    pip install .

Development
~~~~~~~~~~~

For development - clone this repo and run::

    pip install -e ".[dev,docs]"