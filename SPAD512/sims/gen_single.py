import numpy as np
import scipy.optimize as opt
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from skimage.io import imsave, imread
from scipy.signal import convolve, deconvolve
from concurrent.futures import ProcessPoolExecutor, as_completed
from numba import njit
import time

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
            "dark_cps": 0, 
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
        numgates = int(self.freq * self.integ)  # number of repetitions for a single step
        bin_gates = int(numgates / ((2 ** self.bits) - 1))  # number of repetitions per binary image
        data = np.zeros(self.numsteps, dtype=int)  # object to store data
        
        # set up 2D probability array, 1 row for each lifetime
        prob = np.zeros((len(self.lifetimes), len(self.times)))
        for i, lt in enumerate(self.lifetimes):
            lam = 1 / lt
            prob[i, :] += self.zeta * (np.exp(-lam * (self.times)) - np.exp(-lam * (self.times + self.width)))  # based on exponential PDF
            if convolve:
                prob[i, :] = self.convolveProb(prob[i, :]) 

        # optimized binomial drawing for long data using numba
        data += binom_sim(self.bits, len(data), bin_gates, self.weight, prob) # binom_sim is JIT compiled by numba

        dcr = ((self.integ/1e6) * (1/self.dark_cps)) * (self.width * self.freq * 1e-3) # dark count rate based on forward bias open time
        dark_counts = np.random.poisson(dcr, size=self.numsteps) 
        data += dark_counts 

        print(data)

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
        if np.random.random() < 0.001:
            print(f'Generating {pixel}')
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

@njit
def binom_sim(bits, data_len, bin_gates, weight, prob): # does not work for monoexp but later issue i think
    holder = np.zeros(data_len, dtype=np.int16)
    
    for _ in range(2**bits - 1):
        random_vals = np.random.random((data_len, bin_gates))
        choices = random_vals >= weight
        
        for i in range(data_len):
            for j in range(bin_gates):
                choice_index = int(choices[i, j])  # need int for numba
                prob_value = prob[choice_index, i]  
                if np.random.random() < prob_value:  # avoid np.random.binomial(1, prob_value) for numba
                    holder[i] += 1
                    break  # exit inner on first success
    
    return holder