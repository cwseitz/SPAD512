from SPAD512.utils import *
from skimage.io import imread,imsave
import matplotlib.pyplot as plt

config = {
'path': '',
'savepath': '/research2/shared/cwseitz/Data/SPAD/240702/data/intensity_images/',
'prefix': '',
'roi_dim': 256
}

base_prefix = '240702_SPAD-QD-500kHz-100k-1us-1bit-'
base_path = '/research2/shared/cwseitz/Data/SPAD/240702/data/intensity_images/'

acqs = ['acq00003','acq00004','acq00005','acq00006','acq00007']
nums = ['3','4','5','6','7']


for n,acq in enumerate(acqs):
    path = base_path + acq + '/'
    prefix = base_prefix + nums[n]
    config['path'] = path
    config['prefix']  = prefix
    reader = IntensityReaderSparse(config)
    
    #summed = reader.read_1bit_sparse()
    #imsave(config['savepath'] + base_prefix + nums[n] + '-sum.tif',summed)
    #del summed
    
    xmin = 426; xmax = 431; ymin = 357; ymax = 362
    roi = reader.read_1bit_sparse_roi(xmin,xmax,ymin,ymax)
    roi_sum = np.sum(roi,axis=0).astype(np.uint8)
    imsave(config['savepath'] + base_prefix + nums[n] + '-snip2.tif',roi)
    imsave(config['savepath'] + base_prefix + nums[n] + '-snip2-sum.tif',roi_sum)
    del roi; del roi_sum
    
