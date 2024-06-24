from SPAD512.utils import *
from skimage.io import imread,imsave
import matplotlib.pyplot as plt

config = {
'path': '',
'savepath': '/research2/shared/cwseitz/Data/SPAD/240623/data/intensity_images/',
'roi_dim': 256,
'prefix': ''
}

base_prefix = '240623_SPAD-QD-500kHz-50kHz-30k-1bit-'
base_path = '/research2/shared/cwseitz/Data/SPAD/240623/data/intensity_images/'

acqs = ['acq00000','acq00001','acq00002','acq00003'
        'acq00004','acq00005','acq00007',
        'acq00008','acq00009']
nums = ['0','1','2','3','4','5','7','8','9']

acqs = ['acq00006']; nums = ['6']

for n,acq in enumerate(acqs):
    path = base_path + acq + '/'
    prefix = base_prefix + nums[n]
    config['path'] = path
    config['prefix']  = prefix
    #stack = imread(config['savepath'] + base_prefix + nums[n] + '.tif')
    #summed = np.sum(stack,axis=0)
    #plt.imshow(summed)
    #plt.show()
    #imsave(config['savepath'] + base_prefix + nums[n] + '-sum.tif',summed)
    #del stack
    reader = IntensityReader(config)
    stack = reader.stack()
