import os
import numpy as np
import matplotlib.pyplot as plt
import pickle
import sys
import scipy as sc
import scipy.io as sio
import seaborn as sns
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.patches as patches
from pylab import figure, cm
from matplotlib.colors import LogNorm
import matplotlib.colors as colors
from matplotlib.ticker import LogFormatter

import matplotlib
def cmap_map(function, cmap):
    """ Applies function (which should operate on vectors of shape 3: [r, g, b]), on colormap cmap.
    This routine will break any discontinuous points in a colormap.
    """
    cdict = cmap._segmentdata
    step_dict = {}
    # Firt get the list of points where the segments start or end
    for key in ('red', 'green', 'blue'):
        step_dict[key] = list(map(lambda x: x[0], cdict[key]))
    step_list = sum(step_dict.values(), [])
    step_list = np.array(list(set(step_list)))
    # Then compute the LUT, and apply the function to the LUT
    reduced_cmap = lambda step : np.array(cmap(step)[0:3])
    old_LUT = np.array(list(map(reduced_cmap, step_list)))
    new_LUT = np.array(list(map(function, old_LUT)))
    # Now try to make a minimal segment definition of the new LUT
    cdict = {}
    for i, key in enumerate(['red','green','blue']):
        this_cdict = {}
        for j, step in enumerate(step_list):
            if step in step_dict[key]:
                this_cdict[step] = new_LUT[j, i]
            elif new_LUT[j,i] != old_LUT[j, i]:
                this_cdict[step] = new_LUT[j, i]
        colorvector = list(map(lambda x: x + (x[1], ), this_cdict.items()))
        colorvector.sort()
        cdict[key] = colorvector
    return matplotlib.colors.LinearSegmentedColormap('colormap',cdict,1024)

def plot_subplot2(ax, data, data_name, im_min, im_max, i,topbox_loc,arrow_position=[],arrow_set=[],arrow_cor=[],n_dot=2,cmap='hot'):
    # topbox_loc = [92, 161, 15, 15]  # x_start,y_start,x_len,y_len
    sizefont = 15
    xd, yd = 4, 31
    xwid, ywid = 215, 225
    txtcolor = 'w'
    boxedgecolor = 'g'
    boxedgewid = 2
    column = box_stat(data, topbox_loc, n_dot)
    if i == 0:
        plt.text(xd, ywid - yd, column['mutop'] + r'$\pm$' + column['stdtop'], color=txtcolor,
                 fontsize=sizefont)
    elif i == 1:
        plt.text(xd, ywid - yd, '('+column['mutop'] + r'$\pm$' + column['stdtop'] +')'+ 's', color=txtcolor,
                 fontsize=sizefont)
    else:
        plt.text(xd, ywid - yd, '('+column['mutop'] + r'$\pm$' + column['stdtop'] +')'+ 'ms', color=txtcolor,
                 fontsize=sizefont)
        arr = 0
        plt.arrow(arrow_position[arr][0], arrow_position[arr][1], arrow_position[arr][2], arrow_position[arr][3],
                  head_width=arrow_set[0], width=arrow_set[1], ec=arrow_cor[0], fc=arrow_cor[0])
        arr = 1
        plt.arrow(arrow_position[arr][0], arrow_position[arr][1], arrow_position[arr][2], arrow_position[arr][3],
                  head_width=arrow_set[0], width=arrow_set[1], ec=arrow_cor[0], fc=arrow_cor[0])
        arr = 2
        plt.arrow(arrow_position[arr][0], arrow_position[arr][1], arrow_position[arr][2], arrow_position[arr][3],
                  head_width=arrow_set[0], width=arrow_set[1], ec=arrow_cor[1], fc=arrow_cor[1])
        arr = 3
        plt.arrow(arrow_position[arr][0], arrow_position[arr][1], arrow_position[arr][2], arrow_position[arr][3],
                  head_width=arrow_set[0], width=arrow_set[1], ec=arrow_cor[1], fc=arrow_cor[1])

    rect1 = patches.Rectangle((topbox_loc[0], topbox_loc[1]), topbox_loc[2], topbox_loc[3], linewidth=boxedgewid,
                              edgecolor=boxedgecolor, facecolor='none')
    ax.add_patch(rect1)

    ##
    light_hot = cmap_map(lambda x: x * 0.75, matplotlib.cm.hot)
    data = data.T
    a, b = data.shape
    X, Y = np.mgrid[0:a, 0:b]
    Z = data
    pcm = ax.pcolor(X, Y, Z, vmin=im_min, vmax=im_max, cmap=light_hot, shading='auto')
    ax.axis('off')
    return pcm
#
def log_subplot(ax,data,data_name,im_min,im_max,i,topbox_loc,arrow_position=[],arrow_set=[],arrow_cor=[],cmap='hot'):
    ## box stat
    sizefont = 15
    xd, yd = 4, 31
    xwid, ywid = 215, 225
    txtcolor = 'w'
    boxedgecolor = 'g'
    boxedgewid = 2
    column = box_stat(data, topbox_loc, 2)
    if i ==0:
        plt.text(xd, ywid - yd, column['mutop'] + r'$\pm$' + column['stdtop'], color=txtcolor,
                 fontsize=sizefont)
    elif i==1:
        plt.text(xd, ywid - yd, '('+column['mutop'] + r'$\pm$' + column['stdtop']+')'+'s', color=txtcolor,
                 fontsize=sizefont)
    else:
        # plt.text(xd, ywid - yd, column['mutop'] + r'$\pm$' + column['stdtop']+'(ms)', color=txtcolor,
        #          fontsize=sizefont)
        plt.text(xd, ywid - yd, '('+column['mutop'] + r'$\pm$' + column['stdtop'] +')'+ 'ms', color=txtcolor,
                 fontsize=sizefont)
        #add arrow in T2f
        arr = 0
        plt.arrow(arrow_position[arr][0], arrow_position[arr][1], arrow_position[arr][2], arrow_position[arr][3],
                  head_width=arrow_set[0], width=arrow_set[1], ec=arrow_cor[0], fc=arrow_cor[0])
        arr = 1
        plt.arrow(arrow_position[arr][0], arrow_position[arr][1], arrow_position[arr][2], arrow_position[arr][3],
                  head_width=arrow_set[0], width=arrow_set[1], ec=arrow_cor[0], fc=arrow_cor[0])
        arr = 2
        plt.arrow(arrow_position[arr][0], arrow_position[arr][1], arrow_position[arr][2], arrow_position[arr][3],
                  head_width=arrow_set[0], width=arrow_set[1], ec=arrow_cor[1], fc=arrow_cor[1])
        arr = 3
        plt.arrow(arrow_position[arr][0], arrow_position[arr][1], arrow_position[arr][2], arrow_position[arr][3],
                  head_width=arrow_set[0], width=arrow_set[1], ec=arrow_cor[1], fc=arrow_cor[1])

    rect1 = patches.Rectangle((topbox_loc[0], topbox_loc[1]), topbox_loc[2], topbox_loc[3], linewidth=boxedgewid,
                              edgecolor=boxedgecolor, facecolor='none')
    ax.add_patch(rect1)
    ##

    light_hot = cmap_map(lambda x: x * 0.75, matplotlib.cm.hot)
    data = data.T
    a,b=data.shape
    X, Y = np.mgrid[0:a, 0:b]
    Z=data
    Z[Z<0.000001]=0.000001
    pcm = ax.pcolor(X, Y, Z,
                       norm=colors.LogNorm(vmin=im_min, vmax=im_max),
                       cmap=light_hot, shading='auto')
    ax.axis('off')
    return pcm

def box_stat(data,topbox_location,n_dot): # topbox_loc,[xstart,ystart,xlen,ylen]
    boxx = topbox_location[0]
    boxy = topbox_location[1]
    boxxwid = topbox_location[2]
    boxywid = topbox_location[3]

    # 1st column
    column1={}
    column1['boxtop'] = data[ boxy:boxy+boxywid:,boxx:boxx + boxxwid:]
    column1['mutop'] = str(round(np.mean(column1['boxtop']),n_dot))
    column1['stdtop'] = str(round(np.std(column1['boxtop']),n_dot))
    return column1

rootpaper = '/Users/asslaj01/Desktop/work/B0sweeping/mt_results/invivo/20210212_InVivo_MT_v3p2_B0sweep/'
msecrb_mask = 'qM_mid2091_reg1e-05_R_13_svd_basis_v3.2_sweep_0_std_B0_pio2_symmetric_B1_0.9pm0.2_CRBMSE_0511_masked.mat'
mseloss_mask = 'qM_mid2091_reg1e-05_R_13_svd_basis_v3.2_sweep_0_std_B0_pio2_symmetric_B1_0.9pm0.2_MSE_0511_masked.mat'
nlls_mask = 'qM_mid2091_reg1e-05_R_13_svd_basis_v3.2_sweep_0_std_B0_pio2_symmetric_B1_0.9pm0.2'
crb_qm = sio.loadmat(rootpaper+'NN_masked/'+msecrb_mask)['qM']
mse_qm = sio.loadmat(rootpaper+'NN_masked/'+mseloss_mask)['qM']

row = 2
col = 3

##################
### plot each
###################
# figpath='/Users/asslaj01/OneDrive/projects/MRF/MRF_draft/qMaps/nlls_invivo_'
figpath='/Users/asslaj01/OneDrive/projects/MRF_jakob/CRB_manuscript/figures/nlls_invivo_'
c_label = -15
x_bar = 0.85 ##colorbar x postion
yc1 = 0.55 #y position for colorbar
yc2 = 0.06
title_y = 0.9
fig_x = 7.8
fig_y = 5.4
fontsize=16
colorbar_legth=0.4

topbox_loc = [53, 142, 13, 13]  # x_start,y_start,x_len,y_len
#arrow settings
dx, dy = 12,6
arrow_position = [[35,105,dx,dy],[133,105,-dx,dy],[50,85,dx,dy],[121,83,-dx,dy]]
arrow_set =[7,2] #head_width, width
arrow_cor=['green','blue']

# #
for i in range (0,3):
    fig = plt.figure(figsize=(fig_x,fig_y))  # x,y
    n = 1
    if i == 0:
        nlls_name = 'm0s'
        min_v, max_v, min_d, max_d = 0, 0.25, 0, 0.05
        pname = '$m_0^s$'
        nlls_qm = sio.loadmat(rootpaper + 'NLLS_fits_6Dlinapprox/' + nlls_mask)[nlls_name]

        crb_qm_p = crb_qm[:, :, :, i]
        crb = crb_qm_p[85, :, :]
        mse_qm_p = mse_qm[:, :, :, i]
        mse = mse_qm_p[85, :, :]

        nlls_qm = (nlls_qm)
        mse = np.flipud(mse)
        crb = np.flipud(crb)

        ax = plt.subplot(row, col, 1)########## plot NLLS
        plot_subplot2(ax, nlls_qm, 'NLLS', min_v, max_v, i, topbox_loc)

        ax = plt.subplot(row, col, 2)####### plot CRB
        plot_subplot2(ax, crb, 'NLLS', min_v, max_v, i, topbox_loc)
        #
        ax = plt.subplot(row, col, 3)### plot MSE
        colarmap1 = plot_subplot2(ax, mse, 'NLLS', min_v, max_v, i, topbox_loc)
        #
        # # ##################################################### plot difference
        ax = plt.subplot(row, col, 5)
        difference = np.abs(nlls_qm - crb)
        # difference[difference <0.0000001] =-5
        plot_subplot2(ax, difference, '|NLLS-NN|', min_d, max_d, i,topbox_loc,n_dot=3)

        ax = plt.subplot(row, col, 6)
        difference = np.abs(nlls_qm - mse)
        # difference[difference <0.0000001] =-5
        colarmap2 = plot_subplot2(ax, difference, '|NLLS-MSE|', min_d, max_d, i,topbox_loc,n_dot=3)

        cb_ax = fig.add_axes([x_bar, yc1, 0.01, colorbar_legth])
        cbar = fig.colorbar(colarmap1, cax=cb_ax, orientation="vertical")
        cbar.set_ticks([0,0.15,0.3])
        cbar.ax.tick_params(labelsize=fontsize)
        cbar.ax.set_ylabel(pname, labelpad=10, fontsize=fontsize+1)

        cb_ax = fig.add_axes([x_bar, yc2, 0.01,colorbar_legth])
        cbar = fig.colorbar(colarmap2, cax=cb_ax, orientation="vertical")
        cbar.set_ticks([0,0.025,0.5])
        cbar.ax.tick_params(labelsize=fontsize)
        cbar.ax.set_ylabel('$|{m_0^s}_{NN}-{m_0^s}_{NLLS}|$', labelpad=5, fontsize=fontsize+1)

        plt.subplots_adjust(bottom=0, top=0.99, left=0.01, wspace=0.005, hspace=0.005, right=0.84)
        plt.savefig(figpath + nlls_name+'.png', dpi=300)
        plt.show()

    if i == 1:
        nlls_name = 'R1'
        min_v, max_v, min_d, max_d = 1, 4, 0, 0.5
        pname = '$T_1$'
        nlls_qm = sio.loadmat(rootpaper + 'NLLS_fits_6Dlinapprox/' + nlls_mask)[nlls_name]

        crb_qm_p = crb_qm[:, :, :, i]
        crb = crb_qm_p[85, :, :]
        mse_qm_p = mse_qm[:, :, :, i]
        mse = mse_qm_p[85, :, :]

        # nlls_qm = np.flipud(nlls_qm)
        mse = np.flipud(mse)
        crb = np.flipud(crb)

        nlls_qm[nlls_qm < 0.01] = 100000
        nlls_qm = 1 / nlls_qm
        crb = crb
        mse = mse

        ax = plt.subplot(row, col, 1)########## plot NLLS
        log_subplot(ax, nlls_qm, 'NLLS', min_v, max_v, i,topbox_loc)

        ax = plt.subplot(row, col, 2)####### plot CRB
        log_subplot(ax, crb, 'NN-CRB', min_v, max_v, i,topbox_loc)

        ax = plt.subplot(row, col, 3)### plot MSE
        colarmap1 = log_subplot(ax, mse, 'NN-MSE', min_v, max_v, i,topbox_loc)

        # ############ plot difference
        ax = plt.subplot(row, col, 5)
        difference = np.abs(nlls_qm - crb)
        # difference[difference <0.0000001] =-5
        plot_subplot2(ax, difference, '|NLLS-NN|', min_d, max_d, i,topbox_loc,n_dot=3)

        ax = plt.subplot(row, col, 6)
        difference = np.abs(nlls_qm - mse)
        # difference[difference <0.0000001] =-5
        colarmap2 = plot_subplot2(ax, difference, '|NLLS-MSE|', min_d, max_d, i,topbox_loc,n_dot=3)

        cb_ax = fig.add_axes([x_bar, yc1, 0.01, colorbar_legth])
        # cbar = fig.colorbar(colarmap1, cax=cb_ax,ticks=np.unique([1000,4000]), orientation="vertical")
        formatter = LogFormatter(12, labelOnlyBase=False)
        cbar = fig.colorbar(colarmap1, cax=cb_ax, ticks=[1,2,3,4], format=formatter,
                            orientation="vertical")
        cbar.set_ticklabels([1,2,3,4])
        cbar.ax.tick_params(labelsize=fontsize)
        cbar.ax.set_ylabel(pname+'(s)', labelpad=30, fontsize=fontsize+1)

        cb_ax = fig.add_axes([x_bar, yc2, 0.01,colorbar_legth])
        cbar = fig.colorbar(colarmap2, cax=cb_ax, orientation="vertical")
        cbar.set_ticks((0, 0.25, 0.5))
        cbar.ax.tick_params(labelsize=fontsize)
        cbar.ax.set_ylabel('$|{T_1}_{NN}-{T_1}_{NLLS}|$'+'(s)', labelpad=13, fontsize=fontsize+1)

        plt.subplots_adjust(bottom=0, top=0.99, left=0.01, wspace=0.005, hspace=0.005, right=0.84)
        plt.savefig(figpath + nlls_name + '.png', dpi=300)
        plt.show()

    if i == 2:
        nlls_name = 'R2f'
        min_v, max_v, min_d, max_d = 50, 150, 0, 50
        pname = '$T_2^f$'
        nlls_qm = sio.loadmat(rootpaper + 'NLLS_fits_6Dlinapprox/' + nlls_mask)[nlls_name]

        crb_qm_p = crb_qm[:, :, :, i]
        crb = crb_qm_p[85, :, :]
        mse_qm_p = mse_qm[:, :, :, i]
        mse = mse_qm_p[85, :, :]

        nlls_qm = np.flipud(nlls_qm)
        mse = np.flipud(mse)
        crb = np.flipud(crb)

        nlls_qm[nlls_qm < 0.01] = 100000
        nlls_qm = 1 / nlls_qm * 1000
        crb = crb * 1000
        mse = mse * 1000

        row,col=2,3
        ax = plt.subplot(row, col, 1)########## plot NLLS
        log_subplot(ax, nlls_qm, 'NLLS', min_v, max_v, i,topbox_loc, arrow_position,arrow_set,arrow_cor)
        #
        ax = plt.subplot(row, col, 2)####### plot CRB
        colarmap1 = log_subplot(ax, crb, 'NN-CRB', min_v, max_v, i,topbox_loc, arrow_position,arrow_set,arrow_cor)

        ax = plt.subplot(row, col, 3)### plot MSE
        colarmap1 = log_subplot(ax, mse, 'NN-MSE', min_v, max_v, i,topbox_loc, arrow_position,arrow_set,arrow_cor)

        # # ############ plot difference
        ax = plt.subplot(row, col, 5)
        difference = np.abs(nlls_qm - crb)
        # difference[difference <0.0000001] =-5
        plot_subplot2(ax, difference, '|NLLS-NN|', min_d, max_d, i,topbox_loc, arrow_position,arrow_set,arrow_cor)

        ax = plt.subplot(row, col, 6)
        difference = np.abs(nlls_qm - mse)
        # difference[difference <0.0000001] =-5
        colarmap2 = plot_subplot2(ax, difference, '|NLLS-MSE|', min_d, max_d, i,topbox_loc, arrow_position,arrow_set,arrow_cor)

        cb_ax = fig.add_axes([x_bar, yc1, 0.01, colorbar_legth])
        formatter = LogFormatter(10, labelOnlyBase=False)
        cbar = fig.colorbar(colarmap1, cax=cb_ax,ticks=[50,60,90,150], format=formatter,orientation="vertical")
        # cbar = fig.colorbar(colarmap1, cax=cb_ax, orientation="vertical")
        cbar.set_ticklabels([50,60,90,150])
        cbar.ax.tick_params(labelsize=fontsize)
        cbar.ax.set_ylabel(pname+'(ms)', labelpad=8, fontsize=fontsize+1)

        cb_ax = fig.add_axes([x_bar, yc2, 0.01,colorbar_legth])
        cbar = fig.colorbar(colarmap2, cax=cb_ax, orientation="vertical")
        cbar.set_ticks((0, 25,50))
        cbar.ax.tick_params(labelsize=fontsize)
        cbar.ax.set_ylabel('$|{T_2^f}_{NN}-{T_2^f}_{NLLS}|$'+'(ms)', labelpad=18, fontsize=fontsize+1)

        plt.subplots_adjust(bottom=0, top=0.99, left=0.01, wspace=0.005, hspace=0.005, right=0.84)
        plt.savefig(figpath + nlls_name + '.png', dpi=300)
        plt.show()

# text_ax = fig.add_axes([0.01, title_y, 0.3, 0.1])
# text_ax.text(0.5, 0.5, 'NLLS', ha='center', rotation=0, fontsize=fontsize)  # fontweight='semibold')
# text_ax.axis('off')
#
# text_ax = fig.add_axes([0.31, title_y, 0.3, 0.1])
# text_ax.text(0.5, 0.5, 'NN-CRB', ha='center', rotation=0, fontsize=fontsize)  # fontweight='semibold')
# text_ax.axis('off')
#
# text_ax = fig.add_axes([0.61, title_y, 0.3, 0.1])
# text_ax.text(0.5, 0.5, 'NN-MSE', ha='center', rotation=0, fontsize=fontsize)  # fontweight='semibold')
# text_ax.axis('off')

# ##box stat
# column = box_stat(nlls_qm, topbox_loc, 2)
# plt.text(xd, ywid - yd, column['mutop'] + r'$\pm$' + column['stdtop'], color=txtcolor,
#          fontsize=sizefont)
# rect1 = patches.Rectangle((topbox_loc[0], topbox_loc[1]), topbox_loc[2], topbox_loc[3], linewidth=boxedgewid,
#                           edgecolor=boxedgecolor, facecolor='none')
# ax.add_patch(rect1)

# ax = plt.subplot(row, col, 2)  ########## plot NLLS
# data=column['boxtop']
# plot_subplot2(ax, data, 'NLLS', min_v, max_v, i)