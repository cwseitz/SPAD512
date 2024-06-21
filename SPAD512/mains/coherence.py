import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread,imsave
from SPAD512.sr import *

path = ''
file = '240621_SPAD-QD-500kHz-50kHz-30k-400uW-1bit-1-snip1.tif'

stack = imread(path+file)
g2 = G2(stack)
print(g2.shape)
