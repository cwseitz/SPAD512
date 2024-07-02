from SPAD512.utils import *
from skimage.io import imread,imsave
import matplotlib.pyplot as plt

config = {
'path': '',
'savepath': '/research2/shared/cwseitz/Data/SPAD/240630/data/intensity_images/',
'roi_dim': 256,
'prefix': ''
}

base_prefix = '240630_SPAD-QD-500kHz-1k-10ms-8bit-'
base_path = '/research2/shared/cwseitz/Data/SPAD/240630/data/intensity_images/'
acqs = ['acq00000']; nums = ['0']

for n,acq in enumerate(acqs):
    path = base_path + acq + '/'
    prefix = base_prefix + nums[n]
    config['path'] = path
    config['prefix']  = prefix
    reader = IntensityReader(config)
    stack = reader.stack()
