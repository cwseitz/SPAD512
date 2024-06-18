import numpy as np
import matplotlib.pyplot as plt
from binom import PoissonBinomial
from skimage.io import imread

path = '/research2/shared/cwseitz/Data/SPAD/240604/data/intensity_images/'
file = '240604_SPAD-QD-500kHz-50kHz-1us-350uW-1-trimmed-snip5.tif'
stack = imread(path+file)
counts = np.sum(stack,axis=(1,2))
print(np.sum(counts))
print(np.mean(counts))

model = PoissonBinomial(counts,lambd=0.02,zeta_mean=0.05,zeta_std=0.01)
Ns = np.arange(1,20,1)
num_samples=10000
post = [model.integrate(num_samples,n,approx=True) for n in Ns]
post = np.array(post)
post = post/np.sum(post)

fig,ax=plt.subplots(figsize=(3,3))
ax.bar(Ns, post, alpha=0.3, color='red', label=r'$\lambda$='+'0.02 cts')

ax.set_xlim([0,20])
ax.set_xticks(np.arange(0,20,2))
ax.set_xlabel('N')
ax.set_ylabel('Posterior Probability')
ax.legend()
plt.tight_layout()
plt.show()
