from SPAD512.utils import *

config = {
'path': '/research2/shared/cwseitz/Data/SPAD/240618/data/intensity_images/acq00003/',
'savepath': '/research2/shared/cwseitz/Data/SPAD/240618/data/intensity_images/',
'roi_dim': 512,
'prefix': '240618_SPAD-QD-10MHz-1000f-1500uW-1bit'
}

reader = IntensityReader(config)
stack = reader.stack_1bit()
