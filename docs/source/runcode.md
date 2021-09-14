# How to run the code ?


> This short page explains quickly how to install the requirements for this project, and then how to use the code to run simulations.

## Required modules
*First*, install the requirements:
```bash
pip install -r requirements_MRF.txt
```

## Running your own experiments


### [Offline framework](https://mrf.readthedocs.io/en/latest/autodocs/MRF.Offline.html)

#### **STEP 0:** Import the files containing the fingerprints

We need first to import the files containing the data and to save them in the folder *MRF/Offline/loading_data/chosen_name* where *chosen_name* is the name chosen to describe this particular dataset. You should save several files containing the data (typically, I saved around 400 different files each one containing 4096 fingerprints). We need to save different types of files:

-- Files named `fingerprints1.npy`,  `fingerprints2.npy`, etc. Each one of this file should contain a numpy array of size ``(number of fingerprints chosen) x (length of a fingerprint)``. Each fingerprint should be computed solving the Bloch equation for a **proton density equals to 1**.

-- Files named `params1.npy`,  `params2.npy`, etc. Each one of this file should contain a numpy array of size ``(number of fingerprints chosen) x (5)``. Each line should contain the parameters `m_{0s}`, `T_1`, `T_2`, `R` and `T_{2s}` in this order.

-- **If** your loss function required the definition of the Cramer Rao Bound, you should also save files `CRBs1.npy`,  `CRBs2.npy`, etc. Each one of this file should contain a numpy array of size ```(number of fingerprints chosen) x (6)```. Each line should contain the CRBS for the parameters `m_{0s}`, `T_1`, `T_2`, `R`, `T_{2s}` and `PD` **in this order** and considering a **proton density equals to 1**.

*Remark:* You could be disturbed by the fact that the CRBs and the fingerprints are saved with proton densities equal to 1. This is normal since at each epoch, new proton densities are sampled to normalize the loaded fingerprints. The CRBs loaded are also modified to take into account the proton densities sampled.


#### **STEP 1:** Compute a reduced basis for the manifold of the Bloch equation using an online PCA algorithm (optional: basis already computed for you)

**If you use a first linear layer** supposed to perform the projection of the input onto a lower dimensional subspace, you should launch the script `online_PCA.py` located in the folder `online_PCA.py`. This script will compute a reduced basis supposed to approximate the whole manifold of the solutions of the Bloch equation. 

#### **STEP 2:** Compute estimations with the NLLS (optional)

**If you want to be able to use the visualization option** `NLLS VS NN` you should compute the estimations given by the NLLS (Non Linear Least Squares) for a set of hand written parameters. All the details explaining how to do those computations are described in the section `NN VS NLLS and CRB` in the page [visualization of the documentation](https://mrf.readthedocs.io/en/latest/visualize.html).

#### **STEP 3:** Define the desired settings for the training

Open the notebook *offline_training_settings.ipynb* to save easily a pickle file containing all the settings you want. In particular you will be asked to specify a **name** for this settings file. You will use this name to launch your job in the next step.


#### **STEP 4:** Launch your job 

- On your laptop

```bash
python main_offline.py --save_name **chosen_name**
```


- On the NYU cluster *Prince*

Here I give an example of the command line I used to launch my jobs on the Prince cluster of the NYU. I used one GPU to perform the training. 
 
```bash
srun -t160:00:00 --gres=gpu:1 --mem=100GB --jobname=**chosen_name** python main_offline.py --save_name **chosen_name**
```

----


### [Online framework](https://mrf.readthedocs.io/en/latest/autodocs/MRF.Online.html)
#### Define the desired settings for the training

Open the notebook *online_training_settings.ipynb* to save easily a pickle file containing all the settings you want. In particular you will be asked to specify a **name** for this settings file. You will use this name to launch your job in the next step.

#### Compute estimations with the NLLS (optional)

**If you want to be able to use the visualization option** `NLLS VS NN` you should compute the estimations given by the NLLS (Non Linear Least Squares) for a set of hand written parameters. All the details explianing how to do those ocmputations are described in the section `NN VS NLLS and CRB` in the page [visualization of the documentation](https://mrf.readthedocs.io/en/latest/visualize.html).

#### Launch your job

- On your laptop

```bash
python main_online.py --save_name **chosen_name**
```

- On the NYU cluster *Prince*

Here I give an example of the command line I used to launch my jobs on the Prince cluster of the NYU. I used 10 CPUs to perform the training. Note that it's not relevant to launch your job on GPUs since the time required to compute online fingerprints dominate the computational time. 
 
```bash
srun -t160:00:00 --cpus-per-task=10 --mem=10GB --jobname=**chosen_name** python main_online.py --save_name **chosen_name**
```






