import numpy as np
from scipy.ndimage import median_filter
from skimage.feature import blob_log

def compute_sigma(G20,G2m,numm,B):
    dG20 = np.sqrt(G20)
    dG2m = np.sqrt(G2m/numm)
    dB = np.sqrt(B)
    dg20dB = (G20-G2m)/(G2m-B)**2
    dg20dG2m = (G20-B)/(G2m-B)**2
    dg20dG20 = 1/(G2m-B)
    sigma = np.array([dg20dB*dB,dg20dG2m*dG2m,dg20dG20*dG20])
    sigma = np.linalg.norm(sigma)
    return sigma

def coincidence_ratio(stack,dt=1e-3,B=0.0):
    """Expects 5x5 cropped 30k frame sequence"""
    ms = np.arange(1,100,1)
    spac_sum = np.sum(stack,axis=(1,2))
    G20 = np.sum(spac_sum > 1)
    G2ms = []
    for m in ms:
        rolled = np.roll(spac_sum,m)
        G2m = np.sum(spac_sum*rolled >= 1)
        G2ms.append(G2m)
    G2ms = np.array(G2ms)
    G2m = np.mean(G2ms)
    G20 = np.max([G20,B])
    g20 = (G20-B)/(G2m-B)
    sigma = compute_sigma(G20,G2m,len(ms),B)
    return g20,sigma

"""
def coincidence_ratio_batch(stack,patchw=5,plot=False):
    time_sum = np.sum(stack,axis=0)
    med = median_filter(time_sum/time_sum.max(),size=2)
    det = blob_log(med,threshold=0.01,min_sigma=1,max_sigma=5,
                   num_sigma=5,exclude_border=True)
    ndet,_ = det.shape; ratios = []
    for n in range(ndet):
        x,y,_ = det[n]
        x = int(x); y = int(y)
        patch = stack[:,x-patchw:x+patchw,y-patchw:y+patchw]
        r = coincidence_ratio(patch)
        ratios.append(r)
        if plot:
            fig,ax=plt.subplots(1,2)
            ax[0].imshow(patch)
            ax[1].plot(np.sum(patch,axis=(1,2)))
            ax[0].set_title(f'Coincidence ratio: {r}')
            plt.show()
"""

