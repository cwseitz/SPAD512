import numpy as np
from glob import glob
from skimage.io import imsave, imread
from datetime import datetime

'''
Script for making FLIM movies from SPAD acquired data and auto-naming output files
'''

# acquisition parameters for naming; be careful with units
freq = 10 # frequency in MHz
frames = 3 # number of frames
gate_num = 1000 # number of gates per frame
gate_integ = 10 # integration time in ms
gate_width = 5 # gate width in ns
gate_step = 0.018 # gate step size in ns
gate_offset = 0.018 # gate offset in ns
power = 150 # pulsed laser power in uW

# 1-bit acquisiton parameters
bits = 1 # bit depth of data acquisition
globstrs_1bit = ['RAW0000*.bin*'] # filename format for glob to read when stacking 1-bit images

# non-1-bit acquisition parameters
folder = 'acq00001' # folder name with images, no slash at end
roi_dim = 256 # code saves only a square with size roi_dim from the top left of acquisitions

def name(freq, frames, gate_num, gate_integ, gate_width, gate_step, gate_offset, power):
    date =  datetime.now().strftime('%y%m%d')
    filename = f'{date}_SPAD-QD-{freq}MHz-{frames}f-{gate_num}g-{int(gate_integ*1e3)}us-{gate_width}ns-{int(gate_step*1e3)}ps-{int(gate_offset*1e3)}ps-{power}uW.tif'
    return filename

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

def stack_1bit(globstrs, filename):
    for n,globstr in enumerate(globstrs):
        stack = read_bin(globstr,nframes=1000)
        imsave(f'{filename}_stack{n}.tif',stack)

def stack(folder, filename, roi_size):
    files = sorted(glob(f'{folder}/*.png'))
    stack = np.array([imread(f) for f in files])
    imsave(f'{filename}.tif',stack[:,:roi_size,:roi_size])

filename = name(freq, frames, gate_num, gate_integ, gate_width, gate_step, gate_offset, power)
if (bits == 1):
    stack_1bit(globstrs_1bit, filename)
else:
    stack(folder, filename)