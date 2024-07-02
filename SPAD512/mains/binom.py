import numpy as np
import matplotlib.pyplot as plt
from SPAD512.sr import PoissonBinomial
from skimage.io import imread

path = '/research2/shared/cwseitz/Data/SPAD/240630/data/intensity_images/snip2/'
file1bit = '240630_SPAD-QD-500kHz-30k-1us-1bit-1-snip2.tif'
stack1bit = imread(path+file1bit)
counts = np.sum(stack1bit,axis=(1,2))

model = PoissonBinomial(counts,lambd=0.01,zeta_mean=0.01,zeta_std=0.005)
Ns = np.arange(1,20,1)
num_samples=10000
post = [model.integrate(num_samples,n,approx=True) for n in Ns]
post = np.array(post)
post = post/np.sum(post)

fig,ax=plt.subplots(figsize=(3,3))
ax.bar(Ns, post, alpha=0.3, color='red')

ax.set_xlim([0,20])
ax.set_xticks(np.arange(0,20,2))
ax.set_xlabel('N')
ax.set_ylabel('Posterior Probability')
plt.tight_layout()
plt.show()
