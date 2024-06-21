import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread,imsave
from skimage.feature import blob_log
from scipy.ndimage import median_filter
from coherence import *

path = '/Users/cwseitz/Desktop/'
file = '240621_SPAD-QD-500kHz-50kHz-30k-400uW-1bit-1-snip1.tif'

def compute_r(stack):
    spac_sum = np.sum(stack,axis=(1,2))
    rolled = np.roll(spac_sum,1)
    r = np.sum(spac_sum > 1)/np.sum(spac_sum*rolled >= 1)
    r = np.round(r,3)
    return r

stack = imread(path+file)
stack = stack[:,:,:]

time_sum = np.sum(stack,axis=0)
med = median_filter(time_sum/time_sum.max(),size=2)

det = blob_log(med,threshold=0.01,min_sigma=1,max_sigma=5,
               num_sigma=5,exclude_border=True)

patchw = 5
ndet,_ = det.shape
ratios = []
for n in range(ndet):
    x,y,_ = det[n]
    x = int(x); y = int(y)
    r = compute_r(stack[:,x-patchw:x+patchw,y-patchw:y+patchw])
    ratios.append(r)
    fig,ax=plt.subplots(1,2)
    ax[0].imshow(med[x-patchw:x+patchw,y-patchw:y+patchw])
    ax[1].plot(np.sum(stack[:,x-patchw:x+patchw,y-patchw:y+patchw],axis=(1,2)))
    ax[0].set_title(f'Coincidence ratio: {r}')
    plt.show()

#bins = np.linspace(0,1,10)
#plt.hist(ratios,bins=bins)
#plt.show()
