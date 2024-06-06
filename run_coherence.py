import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread,imsave
from skimage.feature import blob_log
from scipy.ndimage import median_filter
from coherence import *

path = '/research2/shared/cwseitz/Data/SPAD/240604/data/intensity_images/'
file = '240604_SPAD-QD-500kHz-50kHz-1us-350uW-1-trimmed.tif'


stack = imread(path+file)
stack = stack[:,:,:]

#g2 = G2(stack)
#g2[g2 < 0.1] = 0.0
#fig,ax=plt.subplots(1,2)
#ax[0].imshow(np.sum(stack,axis=0))
#ax[1].imshow(g2)
#plt.show()
