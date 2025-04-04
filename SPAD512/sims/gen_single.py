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
    def genTrace(self, convolve=False, save=False):
        numgates = int(self.freq * self.integ)  
        bin_gates = int(numgates / ((2 ** self.bits) - 1))  
        total_images = 2 ** self.bits - 1  

        prob = np.zeros(len(self.times))
        for i, lt in enumerate(self.lifetimes):
            lam = 1 / lt
            prob += self.weight[i] * self.zeta * (np.exp(-lam * self.times) - np.exp(-lam * (self.times + self.width)))
        if convolve: prob = self.convolveProb(prob)

        counts_raw = np.random.binomial(numgates, prob)
        P_bin = 1 - (1 - prob) ** bin_gates
        counts = np.random.binomial(total_images, P_bin)

        if save: 
            plt.plot(self.times, counts_raw, label='normal')
            plt.plot(self.times, counts, label='bit')
            plt.legend()
            plt.show()
            np.savez(r'C:\Users\ishaa\Documents\FLIM\figure_remaking\figure1_low',
                    times=self.times,
                    raw=counts_raw,
                    bitted=counts)

        dcr = ((self.integ / 1e6) * (1 / self.dark_cps)) * (self.width * self.freq * 1e-3)
        dark_counts = np.random.poisson(dcr, size=self.numsteps)
        data = counts + dark_counts

        return data.astype(int)
    
    '''Convolution of a probability trace with a Gaussian'''
    def convolveProb(self, trace):
        # set up gaussian IRF
        irf = np.exp(-((self.times - self.irf_mean)**2) / (2 * self.irf_width**2))
        irf /= (self.irf_width * np.sqrt(2*np.pi)) 
        irf /= np.sum(irf) # make sure normalized, this is unnecessary i think it's already normalized

        detected = convolve(trace, irf, mode='full')   

        return detected[:len(trace)] 

    '''Function to help make sure refactoring data generation isn't changing the product'''
    def plotTrace(self, show_max=False, correct=False, save=False, filename='None'):
        data = self.genTrace()
        x = np.arange(len(data)) * self.step + self.offset
        plt.figure(figsize=(6, 4))
        if correct:
            max_counts = 2**self.bits - 1
            probs = data/max_counts
            data = -max_counts * np.log(1 - probs + 1e-9)
            data = 255 * (data/max(data))

        plt.plot(x, data, 'bo', markersize=3, label='Data')
        if save:
            np.savez(filename,
                 x=x,
                 data=data
            )

        # plt.plot(x, decay(x, *params), 'r--', label='Fit: tau = {:.2f}'.format(params[1]))
        if show_max:
            plt.axhline(2**self.bits - 1, color='black', ls='--', label='Max Counts')


        plt.xlabel('Time, ns')
        plt.ylabel('Counts')
        plt.legend()
        plt.grid(True)
        plt.title(f'Simulated Decay for {self.integ*1e-3} ms integration, {self.step} ns step, {self.lifetimes} ns lifetimes')
        plt.show()

        return (x, data)

    '''Helper method for parallelizaiton'''
    def helper(self, pixel):
        # if np.random.random() < 0.002:
        #     print(f'Generating {pixel}')
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

# @njit
# def binom_sim(bits, data_len, bin_gates, weight, prob): # does not work for monoexp but later issue i think
#     holder = np.zeros(data_len, dtype=np.int16)
    
#     for _ in range(2**bits - 1):
#         random_vals = np.random.random((data_len, bin_gates))
#         choices = random_vals >= weight
        
#         for i in range(data_len):
#             for j in range(bin_gates):
#                 choice_index = int(choices[i, j])  # need int for numba
#                 prob_value = prob[choice_index, i]  
#                 if np.random.random() < prob_value:  # avoid np.random.binomial(1, prob_value) for numba
#                     holder[i] += 1
#                     break  # exit inner on first success
    
#     return holder