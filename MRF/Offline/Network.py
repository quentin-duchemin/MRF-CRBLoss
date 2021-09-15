from .Data_class import Data_class
from .Performances import Performances
from ..BaseNetwork import BaseNetwork
from ..Training_parameters import *
import time
import copy
import importlib
from torch import cuda
import os
import torch
from torch import optim
import numpy as np
from numpy import random


class Network(BaseNetwork, Performances):
    """
    Class defining the whole neural network for training.
    The train method will use offline computed fingerprints.
    """

    def __init__(self, name_model, loss, training_parameters, save_name, data_class, validation_settings,
                 projection=None):
        """ New Network."""
        BaseNetwork.__init__(self, name_model, loss, training_parameters, data_class, projection=projection)
        self.save_name = save_name
        Performances.__init__(self, validation_settings)

    def dico_save(self):
        """ Save the results and the settings of the training."""
        dic = Performances.dico_save(self)
        dic.update(BaseNetwork.dico_save(self))
        del dic['trparas']
        del dic['data_class']
        del dic['projection']
        return dic

    def adjust_learning_rate(self, ori_lr, projection_lr_times, optimizer, epoch):
        """Sets the learning rate to the initial LR decayed by 2 every 30 epochs"""
        #lr = max(ori_lr * (0.3 ** (epoch // 3500)), 0.00001)
        lr = max(ori_lr * (0.5 ** (epoch // 40)), 0.00001)
        optimizer.param_groups[0]['lr'] = lr
        #optimizer.param_groups[1]['lr'] = lr * projection_lr_times


    def train(self, lr=0.001, nameoptimizer='Adam',projection_lr_times=1):
        """ Launch the training using the parameter lr as learning rate."""
        if not os.path.exists('save_networks_offline'):
            os.mkdir('save_networks_offline')
        dtype = torch.float
        first_pass = True
        t0 = time.time()
        self.optimizer = nameoptimizer

        ###################################### MODEL
        import importlib
        model = importlib.import_module('MRF.models.' + self.name_model)
        net = model.model(projection=self.projection, nb_params=len(self.trparas.params))
        net = net.to(self.device)
        if self.projection is not None:
            net = self.projection.initialization_first_layer(net, self.device)

        base_params = filter(lambda p: id(p), net.parameters())
        params = [ {'params': base_params} ]

        if nameoptimizer == 'SGD':
            optimizer = optim.SGD(params, lr=lr, momentum=0.9)
        elif nameoptimizer == 'Adam':
            optimizer = optim.Adam(params, lr=lr)

        self.init_validation_B0_complex()

        self.losses_batch = []
        self.losses_train=[]
        for epoch in range(self.trparas.nb_epochs):
            # adjust the learning rate
            self.adjust_learning_rate(lr, projection_lr_times, optimizer, epoch)
            loss_epoch = 0.0
            val_loss_epoch = 0.0
            relative_error = np.zeros(len(self.trparas.params))
            absolute_error = np.zeros(len(self.trparas.params))
            absolute_error_over_CRBs = np.zeros(len(self.trparas.params))
            grad = 0

            iter =0
            for i in range(self.num_files_validation+1, self.data_class.nb_files + 1):
                iter = iter +1
                inputs_file, params_file, CRBs_file = self.data_class.load_data(i)
                inputs_file, PD = self.data_class.proton_density_scaling_B0_complex(inputs_file)
                inputs_file = self.data_class.add_noise_batch_B0(inputs_file)
                PD = PD.reshape(-1, 2) # (b,2)
                PD_norm = PD[:,0]**2 + PD[:,1]**2
                PD_norm = PD_norm.reshape(-1,1)
                CRBs = None
                if self.data_class.CRBrequired:
                    CRBs_file[:, :3] /= np.tile(PD_norm, (1, 3))
                    CRBs_file *= self.data_class.noise_level ** 2
                ndata = inputs_file.shape[0]
                k = 0

                while ((k + 1) * self.trparas.batch_size <= ndata):
                    # zero the parameter gradients
                    net.train()
                    optimizer.zero_grad()
                    inputs = torch.tensor(inputs_file[k * self.trparas.batch_size:(k + 1) * self.trparas.batch_size, :],
                                          dtype=dtype)
                    params = torch.tensor(params_file[k * self.trparas.batch_size:(k + 1) * self.trparas.batch_size, :],
                                          dtype=dtype)
                    inputs = inputs.to(device=self.device)
                    if self.projection is not None: #first layer projection
                        inputs = self.projection.project(inputs)
                    params = params.to(device=self.device)
                    # forward + backward + optimize
                    outputs = net(inputs)
                    if self.data_class.CRBrequired:
                        CRBs = torch.tensor(CRBs_file[k * self.trparas.batch_size:(k + 1) * self.trparas.batch_size,
                                            self.trparas.params], dtype=dtype)
                        CRBs = CRBs.to(device=self.device)

                    loss = self.loss_function(outputs, params, self.trparas.batch_size * len(self.trparas.params),
                                              CRBs=CRBs)
                    loss.backward()
                    optimizer.step()
                    # tracking gradient norm, loss and relative error
                    total_norm = 0
                    for p in net.parameters():
                        param_norm = p.grad.detach().norm(2)
                        total_norm += param_norm.item() ** 2
                    total_norm = total_norm ** (1. / 2)

                    if first_pass:
                        first_pass = False
                        self.trparas.nb_iterations = (ndata * (
                                    self.data_class.nb_files - self.num_files_validation)) // self.trparas.batch_size

                    loss_epoch += loss.detach().item() / self.trparas.nb_iterations
                    relative_error += ((self.compute_relative_errors(outputs.detach(), params,
                                                                     self.trparas.batch_size)).cpu()).numpy() / self.trparas.nb_iterations
                    absolute_error += ((self.compute_absolute_errors(outputs.detach(), params,
                                                                     self.trparas.batch_size)).cpu()).numpy() / self.trparas.nb_iterations

                    if self.data_class.CRBrequired:
                        absolute_error_over_CRBs += ((self.compute_absolute_errors_over_CRBs(outputs.detach(), params,
                                                                                             self.trparas.batch_size,
                                                                                             CRBs)).cpu()).numpy() / self.trparas.nb_iterations
                    grad += total_norm / self.trparas.nb_iterations
                    k += 1

                    # if self.validation:
                    #     net.eval()
                    #     self.dico_validation =  inputs
                    #     if self.data_class.CRBrequired:
                    #         self.CRBs_validation =  torch.tensor(CRBs, dtype=torch.float,device='cpu')
                    #     self.params_validation = torch.tensor(params, dtype=torch.float, device='cpu')
                    #     self.validation_size = self.trparas.batch_size
                    #     self.small_validation_size = 512
                    #     with torch.no_grad():
                    #         if self.projection is not None:
                    #             estimations_validation_graph = net(self.projection.project(self.dico_validation))
                    #         else:
                    #             estimations_validation_graph = net(self.dico_validation)
                    #         estimations_validation = estimations_validation_graph.cpu().detach()
                    #         #self.validation_step(estimations_validation)
                    #
                    #         ### validation
                    #         val_loss = self.loss_function(estimations_validation, self.params_validation,
                    #                                       self.validation_size * len(self.trparas.params),
                    #                                       CRBs=self.CRBs_validation).cpu().detach().numpy()
                    #         val_loss_epoch += val_loss / self.trparas.nb_iterations
                    #
                    #         self.validation_relative_errors.append((self.compute_relative_errors(estimations_validation,
                    #                                                                              self.params_validation,
                    #                                                                              self.validation_size * len(self.trparas.params))).cpu().detach().numpy())
                    #         CRBs_small = self.CRBs_validation[:self.small_validation_size:, :] if self.data_class.CRBrequired else None
                    #         self.losses_small_validation.append((self.loss_function(
                    #             estimations_validation[:self.small_validation_size:,:],
                    #             self.params_validation[:self.small_validation_size:,:],
                    #             self.small_validation_size * len(self.trparas.params), CRBs=CRBs_small)).cpu().detach().numpy())
                    #         self.small_validation_relative_errors.append((self.compute_relative_errors(
                    #             estimations_validation[:self.small_validation_size],
                    #             self.params_validation[:self.small_validation_size],
                    #             self.small_validation_size)).cpu().detach().numpy())
                    #         self.validation_absolute_errors.append((self.compute_absolute_errors(estimations_validation,
                    #                                                                              self.params_validation,
                    #                                                                              self.validation_size * len(self.trparas.params))).cpu().detach().numpy())
                    #         if self.data_class.CRBrequired:
                    #             self.validation_absolute_errors_over_CRBs.append((self.compute_absolute_errors_over_CRBs(
                    #                 estimations_validation,  self.params_validation, self.validation_size * len(self.trparas.params),
                    #                 self.CRBs_validation)).cpu().detach().numpy())
                    #         #print('EPOCH', loss_epoch, 'val:',val_loss,' time ', time.time() - t0)
                    #         ###
            if 1:
                net.eval()
                loss_epoch_separate = 0.0
                with torch.no_grad():
                    number = 0
                    for i in range(self.num_files_validation + 1, self.data_class.nb_files + 1):
                        number = number+1
                        inputs_file, params_file, CRBs_file = self.data_class.load_data(i)
                        inputs_file, PD = self.data_class.proton_density_scaling_B0_complex(inputs_file)
                        inputs_file = self.data_class.add_noise_batch_B0(inputs_file)
                        PD = PD.reshape(-1, 2)  # (b,2)
                        PD_norm = PD[:, 0] ** 2 + PD[:, 1] ** 2
                        PD_norm = PD_norm.reshape(-1, 1)
                        CRBs = None
                        if self.data_class.CRBrequired:
                            CRBs_file[:, :3] /= np.tile(PD_norm, (1, 3))
                            CRBs_file *= self.data_class.noise_level ** 2
                        ndata = inputs_file.shape[0]
                        k = 0

                        inputs = torch.tensor(inputs_file,dtype=dtype)
                        params = torch.tensor(params_file,dtype=dtype)
                        inputs = inputs.to(device=self.device)
                        if self.projection is not None:  # first layer projection
                            inputs = self.projection.project(inputs)
                        params = params.to(device=self.device)
                        outputs = net(inputs)
                        if self.data_class.CRBrequired:
                            CRBs = torch.tensor(CRBs_file[:,self.trparas.params], dtype=dtype)
                            CRBs = CRBs.to(device=self.device)

                        loss = self.loss_function(outputs, params, self.trparas.batch_size * len(self.trparas.params),
                                                  CRBs=CRBs)

                        loss_epoch_separate += loss.detach().item()
            loss_epoch_separate = loss_epoch_separate / (number - 1)
            # if 1:
            #     net.eval()
            #     with torch.no_grad():
            #         outputs = net(inputs)
            #         loss_epoch = self.loss_function(outputs, params, self.trparas.batch_size * len(self.trparas.params),
            #                                         CRBs=CRBs)

            if self.validation:
                net.eval()
                with torch.no_grad():
                    if self.projection is not None:
                        estimations_validation_graph = net(self.projection.project(self.dico_validation))
                    else:
                        estimations_validation_graph = net(self.dico_validation)
                    estimations_validation = estimations_validation_graph.cpu().detach()
                    self.validation_step(estimations_validation)

            print('EPOCH', loss_epoch_separate, 'validation', self.losses_validation[-1], 'time ', time.time() - t0)

            # self.losses_validation.append(val_loss_epoch)
            self.losses.append(loss_epoch_separate)
            self.losses_train.append(loss_epoch)
            print('self.losses_train',self.losses_train[-1])
            self.losses_batch.append(loss.detach().item())
            self.training_relative_errors.append(relative_error)
            self.training_absolute_errors.append(absolute_error)
            if self.data_class.CRBrequired:
                self.training_absolute_errors_over_CRBs.append(absolute_error_over_CRBs)
            self.gradients.append(grad)
            # print('EPOCH', loss_epoch, ' losses_validation ', val_loss_epoch)

            # Saving the results of the training

            dic = {
                'NN': net.state_dict(),
                'learning_rate':  optimizer.param_groups[0]['lr'],
                'time_per_epoch': (time.time() - t0) / (epoch + 1)
            }
            dic.update(self.dico_save())
            torch.save(dic, 'save_networks_offline/network_' + self.save_name)
            if epoch%2==0 and epoch<50:
                torch.save(dic, 'save_networks_offline/network_' + self.save_name+str(epoch))
            elif epoch%4==0 and epoch<100:
                torch.save(dic, 'save_networks_offline/network_' + self.save_name+str(epoch))
            elif epoch %8 == 0 and epoch <150:
                torch.save(dic, 'save_networks_offline/network_' + self.save_name + str(epoch))
            elif epoch %15 == 0 and epoch <200:
                torch.save(dic, 'save_networks_offline/network_' + self.save_name + str(epoch))
            elif epoch %20 == 0 and epoch <250:
                torch.save(dic, 'save_networks_offline/network_' + self.save_name + str(epoch))
            elif epoch %50 == 0 and epoch <350:
                torch.save(dic, 'save_networks_offline/network_' + self.save_name + str(epoch))
            elif epoch % 100 == 0 and epoch < 800:
                torch.save(dic, 'save_networks_offline/network_' + self.save_name + str(epoch))

        print('Training_Finished')
        print('Total_time', time.time() - t0)
