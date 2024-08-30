import numpy as np
import scipy.optimize as opt
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from skimage.io import imsave, imread
from scipy.signal import convolve, deconvolve
from concurrent.futures import ProcessPoolExecutor, as_completed
import random

class Generator:
    def __init__(self,config,**kwargs):
        # default values for parameters when config and keyword args not supplied
        defaults = { 
            "frames": 0,
            "bits": 0,
            "power": 0, 
            "freq": 0,
            "numsteps": 0,
            "integ": 0,
            "width": 0,
            "step": 0,
            "offset": 0,
            "thresh": 0,
            "irf_mean": 0,
            "irf_width": 0, 
            "fit": 0,
            "kernel_size": 0,
            "lifetimes": 0,
            "weight": 0,
            "zeta": 0,
            "x": 0,
            "y": 0,
            "filename": 0,
            "folder": 0
        }

        # update the parameter values based on config then kwargs to prioritize kwargs
        defaults.update(config) 
        defaults.update(kwargs)
        for key, val in defaults.items():
            setattr(self,key,val)

        # update units, ps used in jsons to avoid filename decimals but nanoseconds are numerically easier
        self.step *= 1e-3 # ps --> ns
        self.width *= 1e-3 
        self.offset *= 1e-3

        # auto calculate numsteps if needed based on step size (to fill interpulse period)
        if not self.numsteps:
            self.numsteps = int(1e3 / (self.freq*self.step))
            
        # auto generate filename with standard convention, note unit conversions
        if not self.filename:
            self.filename = (
                f"{config.get('filename', 'default_filename')}_sim-"
                f"{self.freq}MHz-1f-{self.numsteps}g-"
                f"{int(self.integ)}us-{self.width}ns-"
                f"{int((self.step) * 1e3)}ps-{int((self.offset) * 1e3)}ps"
            )

        # create an object for gate opening times to be used in generation and in fitting/plotting
        self.times = (np.arange(self.numsteps) * self.step) + self.offset # ns

    '''Generation of a single fluorescent lifetime trace given the ground truth/SPAD parameters (in self), and convolution requirements'''
    def genTrace(self, convolve=True):
        numgates = int(self.freq * self.integ) # number of repetitions for a single step
        bin_gates = int(numgates / ((2 ** self.bits) - 1)) # number of repetitions per binary image
        data = np.zeros(self.numsteps, dtype=int) # object to store data
        rng = np.random.default_rng() # randomization object
        
        # set up 2d probability array, 1 row for smaller lifetime and another row for the other 
        prob = np.zeros((len(self.lifetimes), len(self.times)))
        for i, lt in enumerate(self.lifetimes):
            lam = 1/lt
            prob[i,:] += self.zeta * (np.exp(-lam * (self.times)) - np.exp(-lam * (self.times + self.width))) # based on exponential PDF 
            if convolve:
                prob[i,:] = self.convolveProb(prob[i,:]) 

        # vectorized generation of binomial data
        choices = rng.choice([0, len(self.lifetimes) - 1], size=(2**self.bits - 1, len(data), bin_gates), p=[self.weight, 1 - self.weight])
        binoms = rng.binomial(1, prob[choices, np.arange(len(data))[None, :, None]])
        successes = np.any(binoms, axis=2)
        data += np.sum(successes, axis=0) # recorded counts are just number of binary frames that recorded a success

        return data

    '''Convolution of a probability trace with a Gaussian'''
    def convolveProb(self, trace):
        # set up gaussian IRF
        irf = np.exp(-((self.times - self.irf_mean)**2) / (2 * self.irf_width**2))
        irf /= (self.irf_width * np.sqrt(2*np.pi)) 
        irf /= np.sum(irf) # make sure normalized, this is unnecessary i think it's already normalized

        detected = convolve(trace, irf, mode='full')   

        return detected[:len(trace)] 

    '''Function to help make sure refactoring data generation isn't changing the product'''
    def plotTrace(self):
        data = self.genTrace()

        x = np.arange(len(data)) * self.step
        plt.figure(figsize=(6, 4))
        plt.plot(x, data, 'bo', markersize=3, label='Data')
        # plt.plot(x, decay(x, *params), 'r--', label='Fit: tau = {:.2f}'.format(params[1]))
        plt.xlabel('Time, ns')
        plt.ylabel('Counts')
        plt.legend()
        plt.title(f'Simulated Decay for {self.integ*1e-3} ms integration, {1e-3*self.step} ns step, {self.lifetimes} ns lifetime')
        plt.show()

    '''Helper method for parallelizaiton'''
    def helper(self, pixel):
        return self.genTrace()

    '''Parallelization coordinator'''
    def genImage(self):
        self.image = np.zeros((self.numsteps, self.x, self.y), dtype=int) # array to store image data

        with ProcessPoolExecutor(max_workers=10) as executor: # max value for max_workers allowed on windows is 60
            futures = {executor.submit(self.helper, (i, j)): (i, j) for i in range(self.x) for j in range(self.y)}

            for future in as_completed(futures):
                i, j = futures[future]
                self.image[:, i, j] = future.result() # store fluorescent trace for the pixel

        # imsave(self.filename + '.tif', self.image)

    