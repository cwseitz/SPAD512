import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread, imsave
import json
import time
from SPAD512.exps import Fitter, Plotter
from SPAD512.sims import Generator, plotLifetimes

'''
Simulation of SPAD time-gated FLIM. Make sure units in .json are consistent with below.

    "freq": MHz, pulsed laser frequency
    "numsteps": counts, number of gate step increments
        - Need to specify 2 or 4 for correct data generation, depending on RLD method
    "integ": us, exposure time for a single gate
        - Need to specify array of values in sim_integs if running simulation on multiple
    "width": ns, width of a single gate
    "step": ps, increment size between gates
        - Need to specify array of values in sim_steps if running simulation on multiple
    "offset": ps, initial increment for first gate step
    "thresh": counts, sum of trace over pixel needed to not discard pixel in analysis
    "kernel_size": counts, number of pixels to take as the kernel size for each pixel (1 --> 3x3 box around each pixel)
    "irf_mean": ns, mean value of gaussian IRF
    "irf_width": ns, stdev of gaussian IRF
    "fit": ('mono', 'mono_conv', 'log_mono_conv', 'mh_mono_conv', 'bi'), analysis method to use
    "lifetimes": ns, lifetimes to simulate (no need to use array if only 1 value)
    "iterations": counts, number of iterations to run
    "zeta": 0<zeta<1, efficiency of photon acquisition
    "x": counts, number of columns to simulate
    "y": counts, number of rows to simulate 
        - x*y gives the total number of iterations for a particular integ/step combo
    "filename": str, name and path to save data with, leave empty for auto generation
'''

config_path = "C:\\Users\\ishaa\\Documents\\FLIM\\SPAD512\\SPAD512\\mains\\run_sim_full.json"
show = True # show final plot

class Simulator:
    def __init__(self, config_path):
        with open(config_path) as f:
            self.config = json.load(f)

    def run_full(self): # array of vals for 'integrations', 'gatesteps', and 'lifetimes' fields in .json
        self.means = np.zeros((len(self.config['lifetimes']),len(self.config['integ']), len(self.config['step'])))
        self.stdevs = np.zeros((len(self.config['lifetimes']),len(self.config['integ']), len(self.config['step'])))
        self.counts = np.zeros((len(self.config['integ']), len(self.config['step'])))
        self.f_vals = np.zeros((len(self.config['integ']), len(self.config['step'])))

        for i, integ in enumerate(self.config['integ']):
            for j, step in enumerate(self.config['step']):
                print(f'Generating data for {(integ * 1e-3):.3f} ms integration, {(step * 1e-3):.3f} ns step')
                tic = time.time()
                dt = Generator(self.config, integ=integ, step=step)
                dt.genImage()
                toc = time.time()
                print(f'Done in {(toc-tic):.1f} seconds, now analyzing.')

                tic = time.time()
                fit = Fitter(self.config, numsteps=dt.numsteps, times=dt.times, step=step, integ=integ)
                results = fit.fit_exps(image=dt.image)
                
                nonzero = results[2][(results[2] != 0) & (~np.isnan(results[2]))]
                self.means[0, i, j] += np.mean(nonzero)
                self.stdevs[0, i, j] += np.std(nonzero)
                self.counts[i,j] += np.mean(results[4])
                self.f_vals[i,j] += self.means[0, i, j]/self.stdevs[0, i, j]

                if len(self.config['lifetimes']) > 1:
                    nonzero = results[3][(results[3] != 0) & (~np.isnan(results[3]))]
                    self.means[1, i, j] += np.mean(nonzero)
                    self.stdevs[1, i, j] += np.std(nonzero)
                    self.f_vals[i,j] += self.means[1,i,j]/self.stdevs[1,i,j]
                
                self.f_vals[i,j] *= np.sqrt(self.counts[i,j])

                toc = time.time()
                print(f'Data analyzed in {(toc-tic):.1f} seconds. F-value {self.f_vals[i,j]:.3f} \n')
                # print(f'Counts: {self.counts[i,j]} \n')

        self.means[np.isnan(self.means)] = 0 
        self.stdevs[np.isnan(self.stdevs)] = 100

        np.savez(self.config['filename'] + '_results.npz', means=self.means, stdevs=self.stdevs, counts=self.counts)
        

    def plot_full(self,show=True):
        results = np.load(self.config['filename'] + '_results.npz')
        self.means = results['means'].astype(float)
        self.stdevs = results['stdevs'].astype(float)
        self.counts = results['counts'].astype(int)
        self.f_vals = results['f_vals'].astype(float)
        
        integs = self.config['integ']
        steps = self.config['step']
        fig, ax = plt.subplots()
        cax = plt.imshow(self.f_vals)
        cbar = fig.colorbar(cax)
        cbar.set_label('Bi-exponential F-values')
        plt.title('F-values for Bi-exponential NNLS at varying integrations and steps')
        ax.set_xlabel('Step size (ns)')
        ax.set_ylabel('Widths (ns)')
        ax.set_yticks(np.linspace(0, integs, num=len(integs), endpoint=False))
        ax.set_yticklabels(integs)
        ax.set_xticks(np.linspace(0, steps, num=len(steps), endpoint=False))
        ax.set_xticklabels(steps)
        plt.setp(ax.get_xticklabels(), rotation=45)

        if show:
            plt.show()

        # plotLifetimes(self.means, self.stdevs, self.config['integ'], self.config['step'], self.config['lifetimes'], self.config['filename'] + '_results', show=show)

if __name__ == '__main__':
    obj = Simulator(config_path)
    obj.run_full()
    obj.plot_full()