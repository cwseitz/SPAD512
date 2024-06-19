from glob import glob
from skimage.io import imread,imsave
import numpy as np

files = sorted(glob('*.png'))
stack = np.array([imread(f) for f in files])
mx = 256
imsave('Stack.tif',stack[:,:mx,:mx])
