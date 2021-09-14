#!/bin/bash

#SBATCH -p cpu_dev
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --array=1
#SBATCH --mem=70G
#SBATCH -t 01:00:00   
#SBATCH --output testing.log        

cd ../../../../notebooks
module add python/gpu/3.6.5
python results_in_vivo_cluster_bloch_1005_complex_PCA.py \
--network_name full_joint_deep3_blochsimv3p2_complexproj_0817_noprojection \
--root /gpfs/data/asslaenderlab/20210715_InVivo_MT_AD_Patient/ \
--num_parameter 3 \
--name_predix nob1_p3_Train_b1_0.21.6_b0_2pi \
--name d0520_v3_nonsweep_varyB0B1_complex_nob1_blochsim_R13_p3_lr_0.01_noise0.01_1000_1024_largerval_train \
--data_file x_mid1496_reg1e-05_R_13_svd_basis_v3.2_sweep_0_std_B0_pio2_symmetric_B1_0.9pm0.2.mat














