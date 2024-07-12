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
    def __init__(self, config, integ=0, step=0, tau=0, numsteps=0,setname=False):
        self.config = config

        self.freq = config['freq']
        self.offset = config['offset'] * 1e-3 # ps --> ns 
        self.width = config['width'] * 1e-3 # ps --> ns
        self.zeta = config['zeta']
        self.x = config['x'] 
        self.y = config['y']

        self.integ = integ if integ else config['integ']
        self.step = step if step else config['step']
        self.step *= 1e-3 # ps --> ns
        self.tau = tau if tau else config['lifetimes']

        if numsteps:
            self.numsteps=numsteps
        else:    
            self.numsteps = int(1e3 / (self.freq*self.step))
        self.config['numsteps']=self.numsteps

        self.times = (np.arange(self.numsteps) * self.step) + self.offset # ns

        self.filename = self.config['filename'] + f'_sim-{self.freq}MHz-1f-{self.numsteps}g-{int(self.integ)}us-{self.width}ns-{int((self.step)*1e3)}ps-{int((self.offset)*1e3)}ps'
        if setname:
            self.config['filename'] = self.filename

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

        choices = (np.random.rand(numgates, self.numsteps) > weight).astype(int)
        events = np.random.rand(numgates, self.numsteps)
        data = np.sum(events < prob[choices, np.arange(self.numsteps)], axis=0)

        plt.plot(self.times, data, 'o')
        plt.show()
        return data

    def convolveProb(self, trace):
        irf_mean = self.config['irf_mean']
        irf_width = self.config['irf_width']
        irf = np.exp(-((self.times - irf_mean)**2) / (2 * irf_width**2))
        irf /= (irf_width * np.sqrt(2*np.pi))

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

    @staticmethod
    def plotLifetimes(mean_image, std_image, integs, steps, tau, savename, show=True):
        numtau, xlen, ylen = np.shape(mean_image)

        # plot mean
        if (numtau == 1):
            fig, ax = plt.subplots(1,2,figsize=(12,6))

            lower = min(max(tau[0] - 5, int(np.min(mean_image[0]))), tau[0] - 1)
            upper = max(min(tau[0] + 5, int(np.max(mean_image[0] + 1))), tau[0] + 1)
            norm = mcolors.TwoSlopeNorm(vmin=lower, vcenter=tau[0], vmax=upper)
            cax1 = ax[0].imshow(mean_image[0], cmap='seismic', norm=norm)
            cbar1 = fig.colorbar(cax1, ax=ax[0], shrink = 0.6)
            cbar1.set_label('Means, ns')
            ax[0].set_title('Mean Lifetimes')
            ax[0].set_xlabel('Step size (ns)')
            ax[0].set_ylabel('Integration time (ms)')
            ax[0].set_yticks(np.linspace(0, xlen, num=xlen, endpoint=False))
            ax[0].set_yticklabels(integs)
            ax[0].set_xticks(np.linspace(0, ylen, num=ylen, endpoint=False))
            ax[0].set_xticklabels(steps)
            plt.setp(ax[0].get_xticklabels(), rotation=45)

            # plot stdevs
            norm = mcolors.TwoSlopeNorm(vmin=-1, vcenter=0, vmax=2)
            cax2 = ax[1].imshow(std_image[0], cmap='seismic', norm=norm)
            cbar2 = fig.colorbar(cax2, ax=ax[1], shrink = 0.6)
            cbar2.set_label('St Devs, ns')
            ax[1].set_title('Standard Deviation of Lifetimes')
            ax[1].set_xlabel('Step size (ns)')
            ax[1].set_ylabel('Integration time (ms)')
            ax[1].set_yticks(np.linspace(0, xlen, num=xlen, endpoint=False))
            ax[1].set_yticklabels(integs)
            ax[1].set_xticks(np.linspace(0, ylen, num=ylen, endpoint=False))
            ax[1].set_xticklabels(steps)
            plt.setp(ax[1].get_xticklabels(), rotation=45)

            plt.savefig(savename, bbox_inches='tight')
            print('Figure saved as ' + savename)

            if show:
                plt.tight_layout()
                plt.show()

        if (numtau == 2):
            fig, ax = plt.subplots(2,2,figsize=(10,10))

            if (np.mean(mean_image[0]) < np.mean(mean_image[1])):
                temp = mean_image[1].copy()
                mean_image[1] = mean_image[0]
                mean_image[0] = temp

                temp = std_image[1].copy()
                std_image[1] = std_image[0]
                std_image[0] = temp

                if (tau[0] < tau[1]):
                    temp = tau[1]
                    tau[1] = tau[0]
                    tau[0] = temp

            lower = min(max(tau[0] - 5, int(np.min(mean_image[0]))), tau[0] - 1)
            upper = max(min(tau[0] + 5, int(np.max(mean_image[0] + 1))), tau[0] + 1)
            norm = mcolors.TwoSlopeNorm(vmin=lower, vcenter=tau[0], vmax=upper)
            cax1 = ax[0, 0].imshow(mean_image[0], cmap='seismic', norm=norm)
            cbar1 = fig.colorbar(cax1, ax=ax[0, 0], shrink = 0.6)
            cbar1.set_label('Means, ns')
            ax[0, 0].set_title('Larger Lifetimes')
            ax[0, 0].set_xlabel('Step size (ns)')
            ax[0, 0].set_ylabel('Integration time (ms)')
            ax[0, 0].set_yticks(np.linspace(0, xlen, num=xlen, endpoint=False))
            ax[0, 0].set_yticklabels(integs)
            ax[0, 0].set_xticks(np.linspace(0, ylen, num=ylen, endpoint=False))
            ax[0, 0].set_xticklabels(steps)
            plt.setp(ax[0, 0].get_xticklabels(), rotation=45)

            lower = min(max(tau[1] - 5, int(np.min(mean_image[1]))), tau[1] - 1)
            upper = max(min(tau[1] + 5, int(np.max(mean_image[1] + 1))), tau[1] + 1)
            norm = mcolors.TwoSlopeNorm(vmin=lower, vcenter=tau[1], vmax=upper)
            cax2 = ax[0, 1].imshow(mean_image[1], cmap='seismic', norm=norm)
            cbar2 = fig.colorbar(cax2, ax=ax[0, 1], shrink = 0.6)
            cbar2.set_label('Means, ns')
            ax[0, 1].set_title('Smaller Lifetimes')
            ax[0, 1].set_xlabel('Step size (ns)')
            ax[0, 1].set_ylabel('Integration time (ms)')
            ax[0, 1].set_yticks(np.linspace(0, xlen, num=xlen, endpoint=False))
            ax[0, 1].set_yticklabels(integs)
            ax[0, 1].set_xticks(np.linspace(0, ylen, num=ylen, endpoint=False))
            ax[0, 1].set_xticklabels(steps)
            plt.setp(ax[0, 1].get_xticklabels(), rotation=45)

            norm = mcolors.TwoSlopeNorm(vmin=-1, vcenter=0, vmax=5)
            cax3 = ax[1, 0].imshow(std_image[0], cmap='seismic', norm=norm)
            cbar3 = fig.colorbar(cax3, ax=ax[1, 0], shrink = 0.6)
            cbar3.set_label('St Devs, ns')
            ax[1, 0].set_title('Standard Deviation of Lifetimes')
            ax[1, 0].set_xlabel('Step size (ns)')
            ax[1, 0].set_ylabel('Integration time (ms)')
            ax[1, 0].set_yticks(np.linspace(0, xlen, num=xlen, endpoint=False))
            ax[1, 0].set_yticklabels(integs)
            ax[1, 0].set_xticks(np.linspace(0, ylen, num=ylen, endpoint=False))
            ax[1, 0].set_xticklabels(steps)
            plt.setp(ax[1, 0].get_xticklabels(), rotation=45)

            norm = mcolors.TwoSlopeNorm(vmin=-1, vcenter=0, vmax=5)
            cax3 = ax[1, 1].imshow(std_image[1], cmap='seismic', norm=norm)
            cbar3 = fig.colorbar(cax3, ax=ax[1, 1], shrink = 0.6)
            cbar3.set_label('St Devs, ns')
            ax[1, 1].set_title('Standard Deviation of Lifetimes')
            ax[1, 1].set_xlabel('Step size (ns)')
            ax[1, 1].set_ylabel('Integration time (ms)')
            ax[1, 1].set_yticks(np.linspace(0, xlen, num=xlen, endpoint=False))
            ax[1, 1].set_yticklabels(integs)
            ax[1, 1].set_xticks(np.linspace(0, ylen, num=ylen, endpoint=False))
            ax[1, 1].set_xticklabels(steps)
            plt.setp(ax[1, 1].get_xticklabels(), rotation=45)

            plt.savefig(savename, bbox_inches='tight')
            print('Figure saved as ' + savename)

            if show:
                plt.tight_layout()
                plt.show()