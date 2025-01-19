import numpy as np
import matplotlib.pyplot as plt

SAVESPEED = 300 # MBps
READOUT = 10 # us
BITDATA = 512*512 # number of bits per 1-bit image

def cal_deadtime(bits, frames, steps): 
    image_data = BITDATA * bits * steps * 1.25e-7 # MB
    image_readout = image_data / SAVESPEED # s
    
    binaries = steps * (2**bits - 1) * READOUT * 1e-6 # s
    return frames * (binaries + image_readout)

def cal_acqtime(frames, steps, integ, bits):
    deatime = cal_deadtime(bits, frames, steps)
    return deatime + frames*steps*integ

# 1 s active time
acq1 = { # fully spatial binning
    'frames': 1,
    'steps': 10,
    'integ': 1,
    'bits': 8,
}

acq2 = { # fully temporal binning
    'frames': 10,
    'steps': 10,
    'integ': 0.1,
    'bits': 8,
}

print(f'Acquisiton 1: {cal_acqtime(**acq1)}')
print(f'Acquisiton 2: {cal_acqtime(**acq2)}')

# 500 ms active time

# 100 ms active time

# 10 ms active time

# 1 ms active time