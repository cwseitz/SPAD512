from SPAD512.utils import *

config = {
'path': '/research2/shared/cwseitz/Data/SPAD/240620/data/intensity_images/acq00000/',
'savepath': '/research2/shared/cwseitz/Data/SPAD/240620/data/intensity_images/',
'roi_dim': 512,
'prefix': '240620_SPAD-QD-10MHz-1000f-1500uW-8bit'
}

reader = IntensityReader(config)
stack = reader.stack()
