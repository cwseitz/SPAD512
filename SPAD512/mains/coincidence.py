import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread,imsave
from skimage.feature import blob_log
from scipy.ndimage import median_filter
from SPAD512.sr import coincidence_ratio

path = '/research2/shared/cwseitz/Data/SPAD/240623/data/intensity_images/'
file = '240623_SPAD-QD-500kHz-50kHz-30k-1bit-9-snip1.tif'

stack = imread(path+file)
stack = stack[:,:,:]
ms = np.arange(1,100,1)
rs = []
for m in ms:
    r = coincidence_ratio(stack,m=m)
    rs.append(r)
rs = np.array(rs)
rs = rs[~np.isinf(rs)]
print(np.mean(rs))

