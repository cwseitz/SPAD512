from SPAD512.utils import *
from skimage.io import imread,imsave
import matplotlib.pyplot as plt

config = {
'freq': 10,
'frames': 1,
'numsteps': 90,
'integ': 50,
'width': 5,
'step': 0.018,
'offset': 0.018,
'power': 500,
'bits': 8,
'folder': '',
'savepath': '/research2/shared/cwseitz/Data/SPAD/240708/data/gated_images/'
}

acqs = [
'50ms1ns'
]

base_path = '/research2/shared/cwseitz/Data/SPAD/240708/data/gated_images/'

for n,acq in enumerate(acqs):
    path = base_path + acq + '/'
    config['folder'] = path
    reader = GatedReader(config)
    reader.stack()
