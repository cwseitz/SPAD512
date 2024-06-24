import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread,imsave
from skimage.feature import blob_log
from scipy.ndimage import median_filter

file = '240623_SPAD-QD-500kHz-50kHz-30k-1bit-9-snip1.tif'

def compute_r(stack,m=1,dt=1e-3):
    spac_sum = np.sum(stack,axis=(1,2))
    nt = len(spac_sum)
    print(np.sum(spac_sum,axis=0))
    t = np.arange(0,nt,1)*dt
    plt.plot(t,spac_sum,color='black')
    plt.xlabel('Time (ms)')
    plt.ylabel('Counts')
    plt.tight_layout()
    plt.show()
    rolled = np.roll(spac_sum,m)
    num_coincident = np.sum(spac_sum > 1) #more than one count in a frame
    num_coincident_seq = np.sum(spac_sum*rolled >= 1) 
    print(num_coincident,num_coincident_seq)
    r = num_coincident/num_coincident_seq
    r = np.round(r,3)
    return r

stack = imread(file)
stack = stack[:,:,:]

r = compute_r(stack)
print(r)
