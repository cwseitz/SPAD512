import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread,imsave
from SPAD512.sr import *

path = '/research2/shared/cwseitz/Data/SPAD/240623/data/intensity_images/'
file = '240623_SPAD-QD-500kHz-50kHz-30k-1bit-8-snip2.tif'

stack = imread(path+file)
nt,nx,ny = stack.shape
stack1 = stack[:,:,:ny//2]
stack2 = stack[:,:,ny//2:]

stack_sum = np.sum(stack,axis=(1,2))
stack1_sum = np.sum(stack1,axis=(1,2))
stack2_sum = np.sum(stack2,axis=(1,2))
stack1_avg = np.mean(stack1_sum)
stack2_avg = np.mean(stack2_sum)

plt.imshow(np.sum(stack,axis=0),vmin=0,vmax=20,cmap='gray')
plt.show()

dt = 1e-3
t = np.arange(0,nt,1)*dt
print(np.sum(stack_sum))
plt.plot(t,stack_sum,color='black')
plt.xlabel('Time (ms)')
plt.ylabel('Counts')
plt.tight_layout()
plt.show()

#idx = np.random.choice(np.arange(0,nt,1),size=10000)
#stack1_sum = stack1_sum[idx]
#stack2_sum = stack2_sum[idx]
#corr = np.correlate(stack1_sum,stack2_sum,mode='full')
#plt.plot(corr)
#plt.show()


