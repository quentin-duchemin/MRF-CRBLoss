# Introduction to the MRF project

As a master student from the [MVA](http://math.ens-paris-saclay.fr/version-francaise/formations/master-mva/), I joined the Center for Data Science of the New York University for my internship from April 2019 to August 2019. Under the supervision of Carlos Fernandez-Granda and Jakob Assländer, I worked on the estimation of the parameters in MRF (Magnetic Resonance Fingerprinting). 

I put on Github the final version of my code with a documentation available [here](https://mrf.readthedocs.io/en/latest/index.html).

### How the repository is organized ?

Two different training strategies can be used:

- using fingerprints that are computed online. In this framework, the network is trained using always new fingerprints computed using the pytorch dataloader which exploits parallelization on different CPUs. 

- using fingerprints pre-computed (offline) and stocked in different files.

### How to launch your own experiments ?

Interactive notebook interfaces have been built to allow the user to easily define the settings desired for the training. These interfaces can be launched opening the jupyter files located in the folder **notebooks**.


```python
git clone https://github.com/quentin-duchemin/MRF.git
cd MRF
sudo pip3 install --upgrade --force-reinstall virtualenv
# create and active the virtualenv
virtualenv pyenv
. pyenv/bin/activate
# install the required python packages in the virtualenv
pip install -r requirements_MRF.txt
# launch the interface online_training_results.ipynb (or offline_training_results.ipynb) and save the settings for your training
cd notebooks
jupyter notebook
# once done, press Ctr C to exit jupyter
# example to launch the training : make sure to replace name_setting_file by the name you gave to the settings file 
python main_online.py --save_name name_setting_file
```

> For further help, you can read the [documentation](https://mrf.readthedocs.io/en/latest/runcode.html).

### How to easily study the results of your training ?

Interactive notebook interfaces have been built to allow the user to easily study the results of the training. These interfaces can be launched opening the files named **offline_visualize_results.ipynb** or **online_visualize_results.ipynb** in the folder **notebooks** using jupyter. 

> You can find a complete description of these notebooks [here](https://mrf.readthedocs.io/en/latest/visualize.html).
