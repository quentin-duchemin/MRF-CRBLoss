import os
from os.path import isfile, join
import importlib
import numpy as np
import torch
import scipy as sc
import scipy.io
import pickle
import sys
sys.path.append('..')
import MRF
from MRF.Training_parameters import *
from MRF.BaseModel import *
from MRF.Projection_complex import *
from MRF.models import *
from MRF.Offline import Network, Data_class, Performances
from MRF.Training_parameters import *
from MRF.simulate_signal import simulation
import matplotlib.pyplot as plt
from scipy.io import loadmat

import argparse

def load_parser():
    parser = argparse.ArgumentParser(description='Description of the training parameters.')
    parser.add_argument('--network_name', type=str, default='full_joint_deep3_blochsimv3p2_0810')
    parser.add_argument('--root', type=str, default='/gpfs/data/asslaenderlab/20191028_Learned_Basis_Functions/20200806_NN_v3p2_v4p2_data')
    parser.add_argument('--name', type=str, default='v6_sweep03_pio2_blochsim_complex_pd_proj_noise_0.005_13D_p3_1002')
    parser.add_argument('--data_file', type=str, default='x_mid2123_reg7e-06_R_13_svd__v6p2_B0sweeping_0p3.mat')
    parser.add_argument('--name_predix', type=str,default='same_as_training')
    parser.add_argument('--compressed', type=int, default=1)
    parser.add_argument('--num_parameter', type=int, default=3)
    return parser


parser = load_parser()
args = parser.parse_args()
number_parameter_predict = args.num_parameter
model = importlib.import_module('MRF.models.' + args.network_name)

save_root=args.root+'/NN_recon/'
if not os.path.exists(save_root):
    os.makedirs(save_root)

print('name:')
print(args.name_predix)
print('root')
print(args.root)
print(args.name_predix)
names = [args.name]

root = args.root


# data_files = ['x_mid1653_reg0_R_9_network_large_R_T2s_v1_noise_0.005_9_dim_CRB_loss_3_parameters_0724.mat',
#              'x_mid1653_reg0_R_9_network_large_R_T2s_v1_pd_scaling_noise_0.005_9_dim_CRB_loss_3_parameters_0724.mat',
#               'x_mid1653_reg0_R_9_network_large_R_T2s_v2_noise_0.005_9_dim_CRB_loss_3_parameters_0724.mat',
#               'x_mid1653_reg0_R_9_network_large_R_T2s_v2_pd_scaling_noise_0.005_9_dim_CRB_loss_3_parameters_0724.mat']

data_files = [args.data_file]

t_1 = time.time()
# total_params_sequence = []
for name, data_file in zip(names, data_files):
    # filepath = 'x_PCA_3.mat'
    # filepath = "/Users/liukangning/downloads/code-MRF-april20/invivo/x_mid1653_reg7e-06_R_10_Kangning_T1.mat"

    filepath = os.path.join(root, data_file)
    with open('../settings_files_offline/settings_'+name+'.pkl', 'rb') as f:
        settings = pickle.load(f)
        net = torch.load(join('../save_networks_offline','network_'+name),map_location='cpu')
        training_parameters = Training_parameters(settings['batch_size'], 1, settings['nb_epochs'], settings['params'], settings['normalization'])
        projection = Projection(settings['start_by_projection'], settings['dimension_projection'], settings['initialization'], settings['normalization'], settings['namepca'])
        data_class = Data_class(training_parameters, settings['noise_type'], settings['noise_level'],
                                           settings['minPD'], settings['maxPD'], settings['nb_files'], settings['path_files'])
        validation_settings = {'validation': settings['validation'],'small_validation_size': settings['small_validation_size'], 'validation_size': settings['validation_size']}
        netw = model.model(projection=projection,nb_params=len(settings['params']))
        device = torch.device('cpu')
        netw.load_state_dict(net['NN'])
        netw.eval()

        import h5py
        import numpy as np
        try:
            from scipy.io import loadmat
            arrays = loadmat(filepath)
            fingers = arrays['x']
            fingers = fingers.T  # (18, 128, 192, 192)
        except:
            arrays = {}
            f = h5py.File(filepath, 'r')
            for k, v in f.items():
                 arrays[k] = np.array(v)
            fingers = arrays['x']
        t_2 = time.time()
        projection.initialization = 'Fixlayer'
        netwproj = model.model(projection=projection,nb_params=len(settings['params']))
        device = torch.device('cpu')
        dico = net['NN']
        if args.compressed:
            print('compressed training: nothing to delete for 1st layer')
        else:
            if net['initialization'] != 'Fixlayer':
                try:
                    del dico['fc1.weight']
                except:
                    del dico['fc1_real.weight']
                    del dico['fc1_imag.weight']
        print(fingers.shape)
        # fingers = fingers[:, 65:95:2, :, :]
        # print(fingers.shape)
        mrfshape = fingers.shape
        netwproj.load_state_dict(dico, strict=False)
        netwproj.eval()
        with torch.no_grad():

            # try:#212   192   160 #256 192 162
            #     fings = fingers.reshape((18,128,192*192))
            # except:
            #     fings = fingers.reshape((9, 128, 192 * 192))
            # print('fingers.shape')
            # print(fingers.shape) #(26, 160, 192, 212)
            fings = fingers.reshape((-1, mrfshape[1],mrfshape[2] * mrfshape[3]))
            sequence_to_stack = []
            for i in range(mrfshape[1]):
                fings_tmp = fings[:,i,:].T
                params_tmp = netwproj(torch.tensor(fings_tmp, dtype=torch.float))
                params_tmp = np.array(params_tmp)
                for ii, para in enumerate(settings['params']):
                    if settings['loss'][para] == 'MSE-Log':
                        params_tmp[:, ii] = 10 ** params_tmp[:, ii]
                params_tmp = params_tmp.reshape((mrfshape[2],mrfshape[3],number_parameter_predict))
                sequence_to_stack.append(params_tmp)
            params = np.stack(sequence_to_stack,axis=0)   # 128*192*192*3 (26, 156, 192, 256) * 3
        processing_time = time.time() - t_2
        params = np.moveaxis(params, [0, 2], [2, 0])
        print(params.shape)  # (215, 215, 170, 3)
        # total_params_sequence.append(params)
        # plt.imshow(params)
        # sc.io.savemat('2params-'+filepath, {'parameters': params})
        sc.io.savemat(os.path.join(save_root, 'qM_'+data_file[2:-4]+ '_'+args.name_predix+'.mat'), {'qM': params})
        total_time = time.time() - t_1
        print(os.path.join(save_root, 'qM_'+data_file[2:-4]+ '_'+args.name_predix+'.mat'))
        print('done results')
        print('processing_time')
        print(processing_time)
        print('total_time')
        print(total_time)
        #
        # with torch.no_grad():
        #
        #     # try:#212   192   160 #256 192 162
        #     #     fings = fingers.reshape((18,128,192*192))
        #     # except:
        #     #     fings = fingers.reshape((9, 128, 192 * 192))
        #     # print('fingers.shape')
        #     # print(fingers.shape) #(26, 160, 192, 212)
        #     fings = -fingers.reshape((-1, mrfshape[1], mrfshape[2] * mrfshape[3]))
        #     sequence_to_stack = []
        #     for i in range(mrfshape[1]):
        #         fings_tmp = fings[:, i, :].T
        #         params_tmp = netwproj(torch.tensor(fings_tmp, dtype=torch.float))
        #
        #         params_tmp = np.array(params_tmp)
        #
        #         params_tmp = params_tmp.reshape((mrfshape[2], mrfshape[3], 3))
        #         sequence_to_stack.append(params_tmp)
        #     params = np.stack(sequence_to_stack, axis=0)  # 128*192*192*3 (26, 156, 192, 256) * 3
        #     params = np.moveaxis(params, [0, 2], [2, 0])
        #     print(params.shape)  # (215, 215, 170, 3)
        #     # total_params_sequence.append(params)
        #     # plt.imshow(params)
        #     # sc.io.savemat('2params-'+filepath, {'parameters': params})
        # sc.io.savemat(os.path.join(save_root, 'qM_'+data_file[:-4] + 'results' + '_' + args.name_predix + '_reverse.mat'),
        #               {'qM': params})
        # print(os.path.join(save_root, 'qM_'+data_file[:-4] + 'results' + '_' + args.name_predix + '_reverse.mat'))
        # print('done results')

# total_params = np.stack(total_params_sequence,axis=3)  # 128*192*192*3
# sc.io.savemat(os.path.join(root, '3params-kangning-m0sT1T2f_0_9.mat' ), {'parameters': total_params})

