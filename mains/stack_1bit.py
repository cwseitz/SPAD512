import numpy as np
import matplotlib.pyplot as plt
from glob import glob
from skimage.io import imsave

def read_bin(globstr,nframes=1000):
	files = glob(globstr)
	stacks = []
	for file in files:
		byte = np.fromfile(file, dtype='uint8')
		bits = np.unpackbits(byte)
		bits = np.array(np.split(bits,nframes))
		bits = bits.reshape((nframes,512,512)).swapaxes(1,2)
		bits = np.flip(bits,axis=1)
		stacks.append(bits)
	stack = np.concatenate(stacks,axis=0)
	return stack

globstrs = ['RAW0000*.bin*']

for n,globstr in enumerate(globstrs):
    stack = read_bin(globstr,nframes=1000)
    imsave(f'Stack{n}.tif',stack)
	


