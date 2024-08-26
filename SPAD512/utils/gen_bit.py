import numpy as np
import scipy.optimize as opt
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from skimage.io import imsave, imread
from scipy.signal import convolve, deconvolve
from concurrent.futures import ProcessPoolExecutor, as_completed
import random

''' 
Simulation of exponential fitting of fluorescent lifetime imaging data acquired by a time-gated SPAD
'''

class Generator:
    def __init__(self,config,**kwargs):
        defaults = {
            'freq': 10,  
            'zeta': 0.05, 
            'x': 1,
            'y': 1,
            'integ': 10000,  
            'lifetimes': [5,20],   
            'width': 15000,
            'offset': 2500,
            'step': 15000,
            'irf_mean': 0,
            'irf_width': 1.4,
            'filename': "",
            'weight': 0.5,
            'numsteps': 4,
            'max_pileups': 5,
            'bits': 8
        }
        defaults.update(config)
        defaults.update(kwargs)

        for key, val in defaults.items():
            setattr(self,key,val)

        self.step *= 1e-3 # ps --> ns
        self.width *= 1e-3 
        self.offset *= 1e-3

        self.rng = np.random.default_rng()

        if not self.numsteps:
            self.numsteps = int(1e3 / (self.freq*self.step))
            
        if not self.filename:
            self.filename = (
                f"{config.get('filename', 'default_filename')}_sim-"
                f"{self.freq}MHz-1f-{self.numsteps}g-"
                f"{int(self.integ)}us-{self.width}ns-"
                f"{int((self.step) * 1e3)}ps-{int((self.offset) * 1e3)}ps"
            )

        self.times = (np.arange(self.numsteps) * self.step) + self.offset # ns

    def genTrace(self, convolve=False, vectorize=False, bit=True):
        numgates = int(self.freq * self.integ) # frequency is only included here i should probably add some more frequency logic
        data = np.zeros(self.numsteps, dtype=int)
        steps = np.arange(self.numsteps) * self.step
        
        prob = np.zeros((len(self.lifetimes), len(steps)))
        for i, lt in enumerate(self.lifetimes):
            lam = 1/lt
            prob[i,:] += self.zeta * (np.exp(-lam * (self.offset + steps)) - np.exp(-lam * (self.offset + steps + self.width)))
            if convolve:
                prob[i,:] = self.convolveProb(prob[i,:])

        bin_gates = int(numgates / ((2 ** self.bits) - 1))
        choices = self.rng.choice([0, len(self.lifetimes) - 1], size=(2**self.bits - 1, len(data), bin_gates), p=[self.weight, 1 - self.weight])
        binoms = self.rng.binomial(1, prob[choices, np.arange(len(data))[None, :, None]])
        successful_detection = np.any(binoms, axis=2)
        data += np.sum(successful_detection, axis=0)

        return data

    def convolveProb(self, trace):
        irf = np.exp(-((self.times - self.irf_mean)**2) / (2 * self.irf_width**2))
        irf /= (self.irf_width * np.sqrt(2*np.pi))

        irf /= np.sum(irf) # discrete normalization

        detected = convolve(trace, irf, mode='full')   

        return detected[:len(trace)] 
    

config = {'dumb': 1}
obj = Generator(config)
print(obj.genTrace())