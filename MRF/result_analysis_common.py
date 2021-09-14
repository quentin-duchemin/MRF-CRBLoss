import matplotlib.pyplot as plt
import torch
import pickle
from os.path import isfile, join

def disp_settings(model_name):
    fullname='../settings_files_offline/settings_'+model_name+'.pkl'
    with open(fullname,'rb') as f:
        sett=pickle.load(f)
    print(sett)

## invivo estimation
def invivo_parameter_estimation(model, model_name, fingerprints):
    parametersm0s = []
    parameterst1 = []
    parameterst2 = []

    with open('../settings_files_offline/settings_' + model_name + '.pkl', 'rb') as f:
        settings = pickle.load(f)
        net = torch.load(join('../save_networks_offline', 'network_' + model_name), map_location='cpu')
        training_parameters = Training_parameters(settings['batch_size'], 1, settings['nb_epochs'], settings['params'],
                                                  settings['normalization'])
        projection = Projection(settings['start_by_projection'], settings['dimension_projection'],
                                settings['initialization'], settings['normalization'], settings['namepca'])
        projection.initialization = 'Fixlayer'
        projection.normalization = 'After_projection'
        data_class = Data_class(training_parameters, settings['noise_type'], settings['noise_level'],
                                settings['minPD'], settings['maxPD'], settings['nb_files'], settings['path_files'])
        validation_settings = {'validation': settings['validation'],
                               'small_validation_size': settings['small_validation_size'],
                               'validation_size': settings['validation_size']}
        netw = model.model(projection=projection, nb_params=len(settings['params']))
        device = torch.device('cuda')
        netw.load_state_dict(net['NN'])
        netw.to(device)

        netw.eval()

    with torch.no_grad():
        fingerprints_tmpt = torch.tensor(fingerprints, dtype=torch.float).to(device)

        prms = netw(fingerprints_tmpt)
        prms = np.array(prms.cpu())
        pars = prms
        for ii, para in enumerate(settings['params']):
            if settings['loss'][para] == 'MSE-Log':
                pars[:, ii] = 10 ** pars[:, ii]
        parametersm0s.append(np.array(pars[:, 0]))
        parameterst1.append(np.array(pars[:, 1]))
        parameterst2.append(np.array(pars[:, 2]))

    parametersm0s = np.array(parametersm0s).reshape((mrfshape[1], mrfshape[2]))
    parameterst1 = np.array(parameterst1).reshape((mrfshape[1], mrfshape[2]))
    parameterst2 = np.array(parameterst2).reshape((mrfshape[1], mrfshape[2]))
    invivo_estimation = np.concatenate((np.expand_dims(parametersm0s, axis=2), np.expand_dims(parameterst1, axis=2),
                            np.expand_dims(parameterst2, axis=2)), axis=2)

    return invivo_estimation

def plot_invivo_slice(invivo_estimation,clorname='jet'):
    fsize=12
    fig = plt.figure(figsize=(21,7))
    ax1=plt.subplot(1, 3, 1)
    plt.imshow(invivo_estimation[:,:,0].T,cmap=clorname,vmin=0, vmax=0.25)
    plt.colorbar()
    plt.title('m0s' , fontsize=10)
    plt.ylabel('Reg7e-06', fontsize=fsize)

    ax1=plt.subplot(1, 3, 2)
    plt.imshow(invivo_estimation[:,:,1].T,cmap=clorname,vmin=0, vmax=2.5)
    plt.colorbar()
    plt.title('T1' , fontsize=10)

    ax1=plt.subplot(1, 3, 3)
    plt.imshow(invivo_estimation[:,:,2].T,cmap=clorname,vmin=0, vmax=0.11)
    plt.colorbar()
    plt.title('T2f' , fontsize=10)
    plt.suptitle(model_name)

def plot_invivo_2_slice_comparison(data_all, model_name, clorname='jet'):
    rwo = 2
    col = 3
    fsize = 12

    data = data_all[0]

    fig = plt.figure(figsize=(21, 14))
    n = 1
    ax1 = plt.subplot(rwo, col, n)
    plt.imshow(data[:, :, 0].T, cmap=clorname, vmin=0, vmax=0.25)
    plt.colorbar()
    plt.title('m0s', fontsize=10)
    plt.ylabel('Reg7e-06', fontsize=fsize)
    n = n + 1

    ax1 = plt.subplot(rwo, col, n)
    plt.imshow(data[:, :, 1].T, cmap=clorname, vmin=0, vmax=2.5)
    plt.colorbar()
    plt.title('T1', fontsize=10)
    n = n + 1

    ax1 = plt.subplot(rwo, col, n)
    plt.imshow(data[:, :, 2].T, cmap=clorname, vmin=0, vmax=0.11)
    plt.colorbar()
    plt.title('T2f', fontsize=10)
    plt.suptitle(model_name[0])
    n = n + 1
    ###########
    data = data_all[1]
    ax1 = plt.subplot(rwo, col, n)
    plt.imshow(data[:, :, 0].T, cmap=clorname, vmin=0, vmax=0.25)
    plt.colorbar()
    plt.title('m0s', fontsize=10)
    plt.ylabel('Reg7e-06', fontsize=fsize)
    n = n + 1

    ax1 = plt.subplot(rwo, col, n)
    plt.imshow(data[:, :, 1].T, cmap=clorname, vmin=0, vmax=2.5)
    plt.colorbar()
    plt.title('T1', fontsize=10)
    n = n + 1

    ax1 = plt.subplot(rwo, col, n)
    plt.imshow(data[:, :, 2].T, cmap=clorname, vmin=0, vmax=0.11)
    plt.colorbar()
    plt.title('T2f', fontsize=10)
    plt.suptitle(model_name[1])
    n = n + 1

    sav = '/gpfs/data/asslaenderlab/share/zhangx19/code-MRF-april20/bias_variance_plot/draft_figure/'
    plt.savefig(sav + 'invi.png', bbox_inches='tight')

## simulated validation estimation
def validation_parameter_estimation(model,model_name, fingerprints):
    parametersm0s = []
    parameterst1 = []
    parameterst2 = []
    with open('../settings_files_offline/settings_'+ model_name+'.pkl', 'rb') as f:
        settings = pickle.load(f)
    net = torch.load(join('../save_networks_offline','network_'+model_name),map_location='cpu')
    training_parameters = Training_parameters(settings['batch_size'], 1, settings['nb_epochs'], settings['params'], settings['normalization'])
    projection = Projection(settings['start_by_projection'], settings['dimension_projection'], settings['initialization'], settings['normalization'], settings['namepca'])
    projection.initialization = 'Fixlayer'
    data_class = Data_class(training_parameters, settings['noise_type'], settings['noise_level'],
                                   settings['minPD'], settings['maxPD'], settings['nb_files'], settings['path_files'])
    validation_settings = {'validation': settings['validation'],'small_validation_size': settings['small_validation_size'], 'validation_size': settings['validation_size']}
    projection.initialization = 'Random'
    netw = model.model(projection=projection,nb_params=len(settings['params']))
    device = torch.device('cuda')
    netw.load_state_dict(net['NN'])
    netw.to(device)

    netw.eval()

    with torch.no_grad():
        for k in range(200):
            fingers = fingerprints
            vec = fingers

            for i in range(fingers.shape[0]):
                noi = np.random.normal(0,1/noise,(R,2))
                fingers[i,:,:] += noi
            fingerprints_tmpt = torch.tensor(fingers, dtype=torch.float).to(device)

            prms = netw(fingerprints_tmpt)
            prms = np.array(prms.cpu())

            pars = prms
            for ii, para in enumerate(settings['params']):
                if settings['loss'][para] == 'MSE-Log':
                    pars[:, ii] = 10 ** pars[:, ii]
            parametersm0s.append(np.array(pars[:, 0]))
            parameterst1.append(np.array(pars[:, 1]))
            parameterst2.append(np.array(pars[:, 2]))

    m0s = np.mean(parametersm0s, axis=0)
    bias_m0s = m0s- parameters[:,0]
    varm0s = np.std(parametersm0s, axis=0) ** 2

    T1 = np.mean(parameterst1, axis=0)
    bias_T1 = T1-parameters[:,1]
    varT1 = np.std(parameterst1, axis=0) ** 2

    T2 = np.mean(parameterst2, axis=0)
    bias_T2 = T2-parameters[:,2]
    varT2 = np.std(parameterst2, axis=0) ** 2
    return m0s,T1,T2,bias_m0s,bias_T1,bias_T2,varm0s,varT1,varT2

def plot_validation(parameters,m0s,T1,T2):
    a=[0,0.5]
    fig = plt.figure(figsize=(15,5))
    ax1=plt.subplot(1, 3, 1)
    plt.plot(parameters[:,0],m0s, 'ro')
    plt.plot(a,a, 'g-')
    plt.title('m0s-predic vs m0s' , fontsize=10)
    plt.xlabel('m0s')
    plt.ylabel('m0s-predict')

    a=[0,5]
    ax1=plt.subplot(1, 3, 2)
    plt.plot(parameters[:,1],T1, 'ro')
    plt.plot(a,a, 'g-')
    plt.title('T1-predic vs T1' , fontsize=10)
    plt.xlabel('T1')
    plt.ylabel('T1-predict')

    a=[0,1.25]
    ax1=plt.subplot(1, 3, 3)
    plt.plot(parameters[:,2],T2, 'ro')
    plt.plot(a,a, 'g-')
    plt.title('T2-predic vs T2' , fontsize=10)
    plt.xlabel('T2')
    plt.ylabel('T2-predict')
    plt.suptitle(model_name)