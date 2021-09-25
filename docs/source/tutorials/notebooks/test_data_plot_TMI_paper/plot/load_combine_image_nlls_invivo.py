import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# path ='/Users/asslaj01/OneDrive/projects/MRF/MRF_draft/qMaps/'
path='/Users/asslaj01/OneDrive/projects/MRF_jakob/CRB_manuscript/figures/'
im1 ='nlls_invivo_m0s'
im2='nlls_invivo_R1'
im3='nlls_invivo_R2f'

row = 3
col = 1

mat='.png'
img1 = mpimg.imread(path+im1+mat)
img2 = mpimg.imread(path+im2+mat)
img3 = mpimg.imread(path+im3+mat)

fig = plt.figure(figsize=(4,8))

ax = plt.subplot(row, col, 1)
imgplot = plt.imshow(img1)
ax.axis('off')
ax = plt.subplot(row, col, 2)
imgplot = plt.imshow(img2)
ax.axis('off')
ax = plt.subplot(row, col, 3)
imgplot = plt.imshow(img3)
ax.axis('off')

fontsize=8
title_y = 0.93
text_ax = fig.add_axes([0.11, title_y, 0.1, 0.1])
text_ax.text(0.5, 0.5, 'NLLS', ha='center', rotation=0, fontsize=fontsize)  # fontweight='semibold')
text_ax.axis('off')

text_ax = fig.add_axes([0.38, title_y, 0.1, 0.1])
text_ax.text(0.5, 0.5, 'NN-CRB', ha='center', rotation=0, fontsize=fontsize)  # fontweight='semibold')
text_ax.axis('off')

text_ax = fig.add_axes([0.65, title_y, 0.1, 0.1])
text_ax.text(0.5, 0.5, 'NN-MSE', ha='center', rotation=0, fontsize=fontsize)  # fontweight='semibold')
text_ax.axis('off')

figpath=path+'/nlls_invivo_'

nlls_name='combined'
plt.subplots_adjust(bottom=0, top=0.98, left=0, wspace=0, hspace=0, right=1)
plt.savefig(figpath + nlls_name + '.png', dpi=600)
plt.show()