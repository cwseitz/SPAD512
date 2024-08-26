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
            'zeta': 5, 
            'x': 0,
            'y': 0,
            'integ': 1,  
            'lifetimes': 10,   
            'width': 0,
            'offset': 0,
            'step': 0,
            'irf_mean': 0,
            'irf_width': 0,
            'filename': "",
            'weight': 0.5,
            'numsteps': 0,
            'max_pileups': 5
        }
        defaults.update(config)
        defaults.update(kwargs)

        for key, val in defaults.items():
            setattr(self,key,val)

        self.step *= 1e-3 # ps --> ns
        self.width *= 1e-3 
        self.offset *= 1e-3

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

    def genTrace(self, convolve=True, vectorize=True, bit=True):
        numgates = int(self.freq * self.integ) # frequency is only included here i should probably add some more frequency logic
        data = np.zeros(self.numsteps, dtype=int)
        steps = np.arange(self.numsteps) * self.step
        
        prob = np.zeros((len(self.lifetimes), len(steps)))
        for i, lt in enumerate(self.lifetimes):
            lam = 1/lt
            prob[i,:] += self.zeta * (np.exp(-lam * (self.offset + steps)) - np.exp(-lam * (self.offset + steps + self.width)))
            if convolve:
                prob[i,:] = self.convolveProb(prob[i,:])

        if bit:
            bin_gates = int(numgates/((2**self.bits)-1))
            counts = 0
            for i in range(len(data)):
                for j in range(2**self.bits - 1):
                    gate_count = 0
                    while((gate_count) < bin_gates) and (data[i] == 0):
                        choice = np.random.choice([0, len(self.lifetimes) - 1], p=[self.weight, 1 - self.weight])
                        data[i] += np.random.Generator.binomial(1, prob[choice, i])



        if vectorize:
            choices = (np.random.rand(numgates, self.numsteps) > self.weight).astype(np.uint8) * (len(self.lifetimes)-1)
            events = np.random.rand(numgates, self.numsteps).astype(np.float32)
            data = np.sum(events < prob[choices, np.arange(self.numsteps)], axis=0)
        else:
            for _ in range(numgates):
                choice = (np.random.rand(self.numsteps) > self.weight).astype(np.int8) * (len(self.lifetimes) - 1)
                event = np.random.rand(self.numsteps).astype(np.float32)
                data += event < prob[choice, np.arange(self.numsteps)]            

        return data
    
    def genTrace_piled(self, convolve=False, weight=0.1):
        numgates = int(self.freq * self.integ)
        data = np.zeros(self.numsteps, dtype=int)
        steps = np.arange(self.numsteps) * self.step
        
        prob = np.zeros((len(self.lifetimes), len(steps), self.max_pileups))
        for i, lt in enumerate(self.lifetimes):
            lam = 1 / lt
            for j in range(self.max_pileups):
                prob[i, :, j] = self.zeta * (
                    np.exp(-lam * (self.offset + steps + j * (1e3 / self.freq)))
                    - np.exp(-lam * (self.offset + steps + j * (1e3 / self.freq) + self.width))
                )
            if convolve:
                for j in range(self.max_pileups):
                    prob[i, :, j] = self.convolveProb(prob[i, :, j])

        choices = (np.random.rand(numgates, self.numsteps) > weight).astype(int) * (len(self.lifetimes) - 1)
        events = np.random.rand(numgates, self.numsteps)
        
        for i in range(self.numsteps): # this is still sus to me i think i vectorized wrong but whatever
            pileup_count = 0
            for k in range(numgates):
                lt_choice = choices[k, i]
                for j in range(self.max_pileups):
                    if events[k, i] < prob[lt_choice, i, j]:
                        data[i] += 1
                        pileup_count += 1
                        break
                if pileup_count >= self.max_pileups:
                    break

        return data

    def convolveProb(self, trace):
        irf = np.exp(-((self.times - self.irf_mean)**2) / (2 * self.irf_width**2))
        irf /= (self.irf_width * np.sqrt(2*np.pi))

        irf /= np.sum(irf) # discrete normalization

        detected = convolve(trace, irf, mode='full')   

        return detected[:len(trace)] 

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

    def helper(self, pixel):
        return self.genTrace()

    def genImage(self):
        self.image = np.zeros((self.numsteps, self.x, self.y), dtype=float)

        with ProcessPoolExecutor(max_workers=10) as executor:
            futures = {executor.submit(self.helper, (i, j)): (i, j) for i in range(self.x) for j in range(self.y)}

            for future in as_completed(futures):
                i, j = futures[future]
                self.image[:, i, j] = future.result()

        # imsave(self.filename + '.tif', self.image)

    