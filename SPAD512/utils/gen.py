import numpy as np
import time
import scipy.optimize as opt
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from skimage.io import imsave, imread
from datetime import datetime
from scipy.signal import convolve, deconvolve
from concurrent.futures import ProcessPoolExecutor, as_completed

''' 
Simulation of exponential fitting of fluorescent lifetime imaging data acquired by a time-gated SPAD
'''

class Generator:
    def __init__(self, config, integ=0, step=0, tau=0):
        self.config = config

        self.freq = config['freq']
        self.offset = config['offset'] * 1e-3 # ps --> ns 
        self.width = config['width']
        self.zeta = config['zeta']
        self.x = config['x'] 
        self.y = config['y']

        self.integ = integ if integ else config['integ']
        self.step = step if step else config['step']
        self.step *= 1e-3 # ps --> ns
        self.tau = tau if tau else config['lifetimes']

        # if self.config['numsteps']:
        #     self.numsteps = self.config['numsteps']
        #     print('yes')
        # else:
        self.numsteps = int(1e3 / (self.freq*self.step))

        print(self.numsteps)

        if self.config['filename']:
            self.filename = self.config['filename']
        else: 
            date =  datetime.now().strftime('%y%m%d')
            self.filename = f'{date}_SPAD-QD-{self.freq}MHz-1f-{self.numsteps}g-{int(self.integ)}us-{self.width}ns-{int((self.step)*1e3)}ps-{int((self.offset)*1e3)}ps-simulated'
            self.config['filename'] = self.filename

        self.times = (np.arange(self.numsteps) * self.step) + self.offset # ns

    def genTrace(self, convolve=True):
        numgates = int(self.freq * self.integ)
        data = np.zeros(self.numsteps, dtype=int)
        steps = np.arange(self.numsteps) * self.step
        
        prob = np.zeros(len(steps))
        for lt in self.tau:
            lam = 1/lt
            prob += self.zeta * (np.exp(-lam * (self.offset + steps)) - np.exp(-lam * (self.offset + steps + self.width)))

        prob /= np.sum(prob)

        for i in range(self.numsteps):
            draws = np.random.rand(numgates) < prob[i]
            data[i] = np.sum(draws)

        if convolve:
            data = self.convolveTrace(data)

        return data

    def convolveTrace(self, trace):
        irf_mean = self.config['irf_mean']
        irf_width = self.config['irf_width']

        irf = np.exp(-((self.times - irf_mean)**2) / (2 * irf_width**2))
        irf /= np.sum(irf)  # normalize

        detected = convolve(trace, irf, mode='full') / np.sum(irf)

        # plt.figure(figsize=(6, 4))
        # plt.plot(self.times, detected[:len(self.times)], 'bo', markersize=3, label='Convolved')
        # plt.plot(self.times, trace, 'ro', markersize=3, label='Original')
        # plt.xlabel('Time, ns')
        # plt.ylabel('Counts')
        # plt.legend()
        # plt.title(f'Simulated exponential/GME curves for 10 ns lifetime, IRF=N(10,0.1)')
        # plt.show()

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
        return self.genTrace(convolve=True)

    def genImage(self):
        self.image = np.zeros((self.numsteps, self.x, self.y), dtype=float)

        with ProcessPoolExecutor() as executor:
            futures = {executor.submit(self.helper, (i, j)): (i, j) for i in range(self.x) for j in range(self.y)}

            for future in as_completed(futures):
                i, j = futures[future]
                self.image[:, i, j] = future.result()

        imsave(self.filename + '.tif', self.image)

    @staticmethod
    def plotLifetimes(mean_image, std_image, integs, steps, tau, show=True):
        numtau, xlen, ylen = np.shape(mean_image)

        fig, ax = plt.subplots(numtau, 2, figsize=(12, 6))
        
        # plot mean
        if (numtau == 1):
            norm = mcolors.TwoSlopeNorm(vmin=np.min(mean_image[0]), vcenter=tau, vmax=np.max(mean_image[0]))
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
            std_image[0] = np.clip(std_image[0], -1, 2)
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

            plt.savefig(str(tau) + 'ns_full', bbox_inches='tight')
            print('Figure saved as ' + str(tau) + 'ns_full.png')

            if show:
                plt.tight_layout()
                plt.show()

        if (numtau == 2):
            if (np.mean(mean_image[0]) < np.mean(mean_image[1])):
                temp = mean_image[1]
                print(mean_image[1])
                mean_image[1] = mean_image[0]
                print(temp)
                mean_image[0] = temp

                temp = std_image[1]
                std_image[1] = std_image[0]
                std_image[0] = temp

                if (tau[0] < tau[1]):
                    temp = tau[1]
                    tau[1] = tau[0]
                    tau[0] = temp
                
            

            norm = mcolors.TwoSlopeNorm(vmin=np.min(mean_image[0]), vcenter=tau[0], vmax=np.max(mean_image[0]))
            cax1 = ax[0, 0].imshow(mean_image[0], cmap='seismic', norm=norm)
            cbar1 = fig.colorbar(cax1, ax=ax[0, 0], shrink = 0.6)
            cbar1.set_label('Means, ns')
            ax[0, 0].set_title('Mean Lifetimes')
            ax[0, 0].set_xlabel('Step size (ns)')
            ax[0, 0].set_ylabel('Integration time (ms)')
            ax[0, 0].set_yticks(np.linspace(0, xlen, num=xlen, endpoint=False))
            ax[0, 0].set_yticklabels(integs)
            ax[0, 0].set_xticks(np.linspace(0, ylen, num=ylen, endpoint=False))
            ax[0, 0].set_xticklabels(steps)
            plt.setp(ax[0, 0].get_xticklabels(), rotation=45)

            norm = mcolors.TwoSlopeNorm(vmin=np.min(mean_image[1]), vcenter=tau[1], vmax=np.max(mean_image[1]))
            cax2 = ax[1, 0].imshow(mean_image[1], cmap='seismic', norm=norm)
            cbar2 = fig.colorbar(cax2, ax=ax[0, 1], shrink = 0.6)
            cbar2.set_label('Means, ns')
            ax[0, 1].set_title('Mean Lifetimes')
            ax[0, 1].set_xlabel('Step size (ns)')
            ax[0, 1].set_ylabel('Integration time (ms)')
            ax[0, 1].set_yticks(np.linspace(0, xlen, num=xlen, endpoint=False))
            ax[0, 1].set_yticklabels(integs)
            ax[0, 1].set_xticks(np.linspace(0, ylen, num=ylen, endpoint=False))
            ax[0, 1].set_xticklabels(steps)
            plt.setp(ax[0, 1].get_xticklabels(), rotation=45)

            std_image[0] = np.clip(std_image[0], -1, 2)
            norm = mcolors.TwoSlopeNorm(vmin=-1, vcenter=0, vmax=2)
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

            std_image[1] = np.clip(std_image[1], -1, 2)
            norm = mcolors.TwoSlopeNorm(vmin=-1, vcenter=0, vmax=2)
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

            plt.savefig(str(tau[0]) + 'ns_' + str(tau[1]) + 'ns_full', bbox_inches='tight')
            print(str(tau[0]) + 'ns_' + str(tau[1]) + 'ns_full.png')

            if show:
                plt.tight_layout()
                plt.show()