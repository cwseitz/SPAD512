import matplotlib.pyplot as plt #type: ignore
from skimage.io import imread #type: ignore
import numpy as np #type: ignore

A = imread('240607/A_6bit.tif')
Intensity = imread('240607/I_6bit.tif')
tau = imread('240607/tau_6bit.tif')

print(tau)
print('max value is')
print(np.amax(tau))

for i in range(len(tau)):
  for j in range(len(tau[0])):
    if tau[i][j] > 1000:
      tau[i][j] = 0
    if tau[i][j] < -1000:
       tau[i][j] = 0

fig,ax=plt.subplots(1,3,figsize=(9,3),sharex=True,sharey=True)
im1 = ax[0].imshow(A,cmap='plasma')
im2 = ax[1].imshow(Intensity,cmap='gray')
im3 = ax[2].imshow(tau,cmap='hsv')
ax[0].set_title('A')
ax[1].set_title('Intensity')
ax[2].set_title(r'$\tau$')

for axi in ax.ravel():
    axi.set_xticks([])
    axi.set_yticks([])

plt.colorbar(im1,ax=ax[0],label='cts')
plt.colorbar(im2,ax=ax[1],label='cts')
plt.colorbar(im3,ax=ax[2],label='ns')
plt.tight_layout()
plt.show()
