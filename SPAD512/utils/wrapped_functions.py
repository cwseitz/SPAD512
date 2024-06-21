# stack_gated.py
from glob import glob
from skimage.io import imread,imsave
import numpy as np

files = sorted(glob('*.png'))
stack = np.array([imread(f) for f in files])
mx = 256
imsave('Stack.tif',stack[:,:mx,:mx])



# stack_8bit.py
from glob import glob
from skimage.io import imread,imsave
import numpy as np

files = sorted(glob('*.png'))
stack = np.array([imread(f) for f in files])
mx = 256
imsave('Stack.tif',stack[:,:mx,:mx])



# stack_1bit.py
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
	


# name.py
from datetime import datetime

freq = 10 # frequency in MHz
frames = 3 # number of frames
gate_num = 1000 # number of gates per frame
gate_integ = 10 # integration time in ms
gate_width = 5 # gate width in ns
gate_step = 0.018 # gate step size in ns
gate_offset = 0.018 # gate offset in ns
power = 150 # pulsed laser power in uW

def name(freq, frames, gate_num, gate_integ, gate_width, gate_step, gate_offset, power):
    date =  datetime.now().strftime('%y%m%d')
    filename = f'{date}_SPAD-QD-{freq}MHz-{frames}f-{gate_num}g-{int(gate_integ*1e3)}us-{gate_width}ns-{int(gate_step*1e3)}ps-{int(gate_offset*1e3)}ps-{power}uW.tif'
    return filename

filename = name(freq, frames, gate_num, gate_integ, gate_width, gate_step, gate_offset, power)
print(filename)


