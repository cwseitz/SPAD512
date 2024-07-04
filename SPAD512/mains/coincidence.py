import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread,imsave
from skimage.feature import blob_log
from scipy.ndimage import median_filter
from scipy import stats
from glob import glob
from SPAD512.sr import coincidence_ratio

def concatenate(files):
    stacks = [imread(f) for f in files]
    return np.concatenate(stacks,axis=0)

path = '/research2/shared/cwseitz/Data/SPAD/240702/data/intensity_images/snip2/'
globstr = '240702_SPAD-QD-500kHz-100k-1us-1bit-*-snip2.tif'
file8bit = '240702_SPAD-QD-500kHz-1k-10ms-8bit-1-snip2.tif'
files1bit = glob(path+globstr)

stack1bit = concatenate(files1bit)
stack8bit = imread(path+file8bit)
counts = np.sum(stack1bit,axis=(1,2))
g20,sigma = coincidence_ratio(stack1bit,B=37.5)
threshold = 0.5
conf = stats.norm.cdf(threshold,loc=g20, scale=sigma)

g20 = np.round(g20,2); sigma = np.round(sigma,2); conf = np.round(conf,2)
t1 = np.linspace(0,500,500000)
t2 = np.linspace(0,10,999)
fig,ax=plt.subplots(1,3,figsize=(10,3))
ax[0].imshow(np.sum(stack1bit,axis=0),cmap='gray')
ax[0].set_title('ROI (' + str(np.sum(counts)) + ' cts)')
ax[1].plot(t1,counts,color='black')
ax[2].plot(t2,np.sum(stack8bit,axis=(1,2)),color='gray')
valstr = r'$g^{(2)}(0)=$' + str(g20) +\
         r' $\sigma=$' + str(sigma) + f' confidence={conf}'
ax[1].set_title(valstr)
ax[1].set_xlabel('Time (ms)')
ax[1].set_ylabel('cts')
ax[2].set_xlabel('Time (s)')
ax[2].set_ylabel('ROI Average cts')
plt.tight_layout()
plt.show()

