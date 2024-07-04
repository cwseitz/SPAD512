import numpy as np
import matplotlib.pyplot as plt
from SPAD512.sr import PoissonBinomial2
from skimage.io import imread
from glob import glob

def concatenate(files):
    stacks = [imread(f) for f in files]
    return np.concatenate(stacks,axis=0)

path = '/research2/shared/cwseitz/Data/SPAD/240702/data/intensity_images/snip1/'
globstr = '240702_SPAD-QD-500kHz-100k-1us-1bit-*-snip1.tif'
files1bit = glob(path+globstr)
stack1bit = concatenate(files1bit)
counts = np.sum(stack1bit,axis=(1,2))

model = PoissonBinomial2(counts,lambd=0.0075,zeta_mean=0.01,zeta_std=0.005)
Ns = np.arange(1,20,1)
num_samples=1000
post = [model.integrate(num_samples,n) for n in Ns]
post = np.array(post)
post = post/np.sum(post)
print(post)

fig,ax=plt.subplots(figsize=(3,3))
ax.bar(Ns, post, alpha=0.3, color='red')

ax.set_xlim([0,20])
ax.set_xticks(np.arange(0,20,2))
ax.set_xlabel('N')
ax.set_ylabel('Posterior Probability')
plt.tight_layout()
plt.show()
