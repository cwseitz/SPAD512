from SPAD512.utils import *
from skimage.io import imread,imsave
import matplotlib.pyplot as plt

config = {
'path': '',
'savepath': '/research2/shared/cwseitz/Data/SPAD/240630/data/intensity_images/',
'roi_dim': 256,
'prefix': ''
}

base_prefix = '240630_SPAD-QD-500kHz-30k-1us-1bit-'
base_path = '/research2/shared/cwseitz/Data/SPAD/240630/data/intensity_images/'

acqs = ['acq00001','acq00002','acq00003','acq00004','acq00005']
nums = ['1','2','3','4','5']


for n,acq in enumerate(acqs):
    path = base_path + acq + '/'
    prefix = base_prefix + nums[n]
    config['path'] = path
    config['prefix']  = prefix
    reader = IntensityReader(config)
    stack = reader.stack_1bit()
    summed = np.sum(stack,axis=0)
    summed = summed.astype(np.uint8)
    imsave(config['savepath'] + base_prefix + nums[n] + '-sum.tif',summed)
    del stack
