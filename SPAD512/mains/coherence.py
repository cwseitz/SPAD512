import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread,imsave
from SPAD512.sr import *

path = '/research2/shared/cwseitz/Data/SPAD/240623/data/intensity_images/'
file = '240623_SPAD-QD-500kHz-50kHz-30k-1bit-9-snip1.tif'

stack = imread(path+file)
#stack = np.zeros((30000,10,10), dtype=int)
#random_heights = np.random.randint(0,10,size=30000)
#random_widths = np.random.randint(0,10,size=30000)
#stack[np.arange(30000),random_heights,random_widths] = 1

nt,nx,ny = stack.shape
stack1 = stack[:,:,:ny//2]
stack2 = stack[:,:,ny//2:]

stack1_sum = np.sum(stack1,axis=(1,2))
stack2_sum = np.sum(stack2,axis=(1,2))
stack1_avg = np.mean(stack1_sum)
stack2_avg = np.mean(stack2_sum)

plt.plot(stack1_sum*stack2_sum)
plt.show()

#idx = np.random.choice(np.arange(0,nt,1),size=10000)
#stack1_sum = stack1_sum[idx]
#stack2_sum = stack2_sum[idx]
corr = np.correlate(stack1_sum,stack2_sum,mode='full')
plt.plot(corr)
plt.show()


