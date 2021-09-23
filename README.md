# Presentation of the repository

This repository contains the *MRF-CRBLoss* package which allows to estimate biological parameters in the context of Magnetic Resonance Fingerprinting (MRF). Our main contribution lies in the introduction of a new loss function that extends the usual MSE by normalizing the squared errors for each parameter by its respective Cramer Rao Bound. 

In our paper, we prove that our approach allows to improve estimation results. As a by product, this new loss function provides an absolute metric to quantify the performance of the network.

In the documentation of this package (https://quentin-duchemin.github.io/MRF-CRBLoss/), we provide all ingredients allowing to reproduce the experiments of our paper. Moreover, we describe in details our procedure allowing reseachers to fork our work to use or extend our code for their own applications.



# Reference

If you used MRF-CRBLoss in your research, please consider citing

```
@misc{zhang2021cramerrao,
      title={Cram\'er-Rao bound-informed training of neural networks for quantitative MRI}, 
      author={Xiaoxia Zhang and Quentin Duchemin and Kangning Liu and Sebastian Flassbeck and Cem Gultekin and Carlos Fernandez-Granda and Jakob Assländer},
      year={2021},
      eprint={2109.10535},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```
