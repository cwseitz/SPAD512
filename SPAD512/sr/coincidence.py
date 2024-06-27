import numpy as np
from scipy.ndimage import median_filter
from skimage.feature import blob_log

def coincidence_ratio(stack,m=1,dt=1e-3):
    spac_sum = np.sum(stack,axis=(1,2))
    nt = len(spac_sum)
    t = np.arange(0,nt,1)*dt
    rolled = np.roll(spac_sum,m)
    num_coincident = np.sum(spac_sum > 1)
    num_coincident_seq = np.sum(spac_sum*rolled >= 1) 
    r = num_coincident/num_coincident_seq
    r = np.round(r,3)
    return r

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
            ax[0].imshow(med[x-patchw:x+patchw,y-patchw:y+patchw])
            ax[1].plot(np.sum(patch,axis=(1,2)))
            ax[0].set_title(f'Coincidence ratio: {r}')
            plt.show()

