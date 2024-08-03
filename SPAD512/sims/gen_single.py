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
    def __init__(self,config,integ=0,step=0,tau=0,numsteps=0,width=0,offset=0,freq=0,zeta=0,x=0,y=0,irf_mean=0,irf_width=0,filename=""):
        self.freq = freq if freq else config['freq']
        self.zeta = zeta if zeta else config['zeta']
        self.x = x if x else config['x'] 
        self.y = y if y else config['y']
        self.integ = integ if integ else config['integ']
        self.tau = tau if tau else config['lifetimes']
        self.width = width if width else config['width']
        self.offset = offset if offset else config['offset']
        self.step = step if step else config['step']
        self.irf_mean = irf_mean if irf_mean else config['irf_mean']
        self.irf_width = irf_width if irf_width else config['irf_width']

        self.step *= 1e-3 # ps --> ns
        self.width *= 1e-3 
        self.offset *= 1e-3

        if numsteps:
            self.numsteps=numsteps
        else:    
            self.numsteps = int(1e3 / (self.freq*self.step))

        self.times = (np.arange(self.numsteps) * self.step) + self.offset # ns

        self.filename = filename if filename else (
            self.config['filename'] + f'_sim-{self.freq}MHz-1f-{self.numsteps}g-'
            f'{int(self.integ)}us-{self.width}ns-{int((self.step)*1e3)}ps-'
            f'{int((self.offset)*1e3)}ps'
        )

    def genTrace(self, convolve=False, weight=0.1):
        numgates = int(self.freq * self.integ)
        data = np.zeros(self.numsteps, dtype=int)
        steps = np.arange(self.numsteps) * self.step
        
        prob = np.zeros((len(self.tau), len(steps)))
        for i, lt in enumerate(self.tau):
            lam = 1/lt
            prob[i,:] += self.zeta * (np.exp(-lam * (self.offset + steps)) - np.exp(-lam * (self.offset + steps + self.width)))
            if convolve:
                prob[i,:] = self.convolveProb(prob[i,:])

        choices = (np.random.rand(numgates, self.numsteps) > weight).astype(int) * (len(self.tau)-1)
        events = np.random.rand(numgates, self.numsteps)
        data = np.sum(events < prob[choices, np.arange(self.numsteps)], axis=0)

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
        plt.title(f'Simulated Decay for {self.integ*1e-3} ms integration, {1e-3*self.step} ns step, {self.tau} ns lifetime')
        plt.show()

    def helper(self, pixel):
        return self.genTrace()

    def genImage(self):
        self.image = np.zeros((self.numsteps, self.x, self.y), dtype=float)

        with ProcessPoolExecutor() as executor:
            futures = {executor.submit(self.helper, (i, j)): (i, j) for i in range(self.x) for j in range(self.y)}

            for future in as_completed(futures):
                i, j = futures[future]
                self.image[:, i, j] = future.result()

        # imsave(self.filename + '.tif', self.image)

    