Presentation of the project
===========================

What is Magnetic Resonance Fingerprinting (MRF)?
------------------------------------------------

By considering radio frequency pulses design in a pseudo-random fashion, MRF aims at triggering signal responses that are intrinsic to each tissue. As consequence, the signal response obtained from a specific voxel is called a *fingerprint*. 

While signals obtained from traditional MRI scans can usually by analyze to extract information on a single biophysical parameter, MRF allows to acquire signals that carry the tissue properties. As a result, MRF leads to a time saving regarding the scanning process and allows to obtain a quantitative information on the tissue properties, while scan analysis of traditional MRI techniques are qualitative.


What are the challenges for MRF?
--------------------------------

The previous paragraph shed light on the disruptive advantages that MRF could provide over actual methods. Nevertheless, with MRF techniques arise new scientific difficulties that are feeding a large span of current research. The greatest challenge in MRF is to be able for a given *fingerprint* to efficiently obtain the corresponding tissue properties.

First approaches to tackle this problem are based on a precomputed dictionary of fingerprints. Then, one could simply use template matching techiques to estimate the tissue's biomarkers associated to a fingerprint. These methods can be computationally heavy and one way to improve their execution time is to use a low rank approximation of the dictionary (by computing its SVD) and to use compressed sensing techniques. Despite this effort, dictionary based methods cannot allow to solve the problem at stake since they are intrinsically limited by the curse of dimensionality. 

As a consequence, dictionary free approaches have been inquired and ones of the most promising are deep-learning methods. 

What are our main contributions?
--------------------------------

A high dimensional biological method
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

We use an 8-parameter magnetization transfer model :doc:`Asslander19a <references>`. It is based on Henkelman’s original two-pool spin model :doc:`Henkelman93 <references>` that distinguishes between protons bound in water—the so-called free pool—and protons bound in macromolecules, such as proteins or lipids—the so-called semi-solid pool. The pulse sequence is designed such that the free pool remains in the hybrid state :doc:`Asslander19a, Asslander19b <references>`, —a spin ensemble state that provides a combination of robust and tractable spin dynamics with the ability to encode biophysical parameters with high signal-to-noise ratio (SNR) efficiency compared to steady-state MR experiments :doc:`Asslander19b <references>`.

A new loss function
^^^^^^^^^^^^^^^^^^^

Our main contribution lies in the definition of a new loss function. While most of existing works use the Mean Squared Error (MSE) as loss function, we propose to normalize each squared difference by the corresponding Cramer Rao Bound. This loss has the advantage of automatically scaling the contribution of each sample in a batch according to the difficulty to estimate the biomarker in the region of interest in the parameter space. More precisely, parameter hard-to-estimate will see their contribution decreased in the overall gradient computation, avoiding to see the loss domines by such hard-to-estimate samples. Further, our CRBbased loss function gives an absolute metric to measure the performance of the trained network. Indeed, considering that the network is unbiased, the overall loss function is theoretically ensured to be larger than one. The closer the loss is to one, the closer the neural network is to a maximum efficiency unbiased estimator.


A well documented and maintainable code
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

With this :doc:`documentation <documentation/index>` and our :doc:`tutorials <tutorials/index>`, one can easily reproduce the results of our paper. Moreover, we put a lot of effort to make our code extensible and flexible. We give a detailed description of the structure of the code in order to allow reseachers to use our code for their own projects. We give some tips to extend our package :doc:`here <developer/index>`.

.. toctree::
    :maxdepth: 1
    :titlesonly:
    :hidden:

    installation.rst
    documentation/index.rst
    tutorials/index.rst
    developer/index.rst
    references.rst
