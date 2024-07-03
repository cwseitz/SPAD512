from SPAD512.utils import *
from skimage.io import imread,imsave
import matplotlib.pyplot as plt

config = {
'path': '',
'savepath': '/research2/shared/cwseitz/Data/SPAD/240702/data/intensity_images/',
'roi_dim': 512,
'prefix': ''
}

base_prefix = '240702_SPAD-QD-500kHz-1k-10ms-8bit-'
base_path = '/research2/shared/cwseitz/Data/SPAD/240702/data/intensity_images/'
acqs = ['acq00001','acq00002']; nums = ['1','2']

for n,acq in enumerate(acqs):
    path = base_path + acq + '/'
    prefix = base_prefix + nums[n]
    config['path'] = path
    config['prefix']  = prefix
    reader = IntensityReader(config)
    stack = reader.stack()
