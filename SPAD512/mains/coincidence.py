import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread,imsave
from skimage.feature import blob_log
from scipy.ndimage import median_filter
from scipy import stats
from SPAD512.sr import coincidence_ratio

path = '/research2/shared/cwseitz/Data/SPAD/240630/data/intensity_images/snip1/'
file1bit = '240630_SPAD-QD-500kHz-30k-1us-1bit-2-snip1.tif'
file8bit = '240630_SPAD-QD-500kHz-1k-10ms-8bit-0-snip1.tif'

stack1bit = imread(path+file1bit)
stack8bit = imread(path+file8bit)
counts = np.sum(stack1bit,axis=(1,2))
g20,sigma = coincidence_ratio(stack1bit,B=3.0)
threshold = 0.5
conf = stats.norm.cdf(threshold,loc=g20, scale=sigma)

g20 = np.round(g20,2); sigma = np.round(sigma,2); conf = np.round(conf,2)
t1 = np.linspace(0,30,30000)
t2 = np.linspace(0,10,1000)
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

