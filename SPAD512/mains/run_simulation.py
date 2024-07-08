import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread, imsave
import json
import time

from SPAD512.exps import Fitter, Plotter
from SPAD512.utils import Generator

'''
Simulation of SPAD time-gated FLIM. Make sure units in .json are consistent with below.

    "freq": MHz, pulsed laser frequency
    "numsteps": counts, number of gate step increments
    "integ": us, exposure time for a single gate
    "width": ns, width of a single gate
    "step": ps, increment size between gates
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
        Note: x*y gives the total number of trials for a particular integ/step/lifetime combo, if running more than 1 set of 3.
    "filename": str, name and path to save data with, leave empty for auto generation
'''

config_path = 'SPAD512/SPAD512/mains/run_simulation.json'
show = True # show final plot
sim_steps = [18, 40]
sim_integs = [2500, 5000]


class Simulator:
    def __init__(self, config_path):
        with open(config_path) as f:
            self.config = json.load(f)
    
    def run_full(self, sim_steps, sim_integs): # array of vals for 'integrations', 'gatesteps', and 'lifetimes' fields in .json
        self.means = np.zeros((len(self.config['lifetimes']),len(sim_integs), len(sim_steps)))
        self.stdevs = np.zeros((len(self.config['lifetimes']),len(sim_integs), len(sim_steps)))

        for i, integ in enumerate(sim_integs):
            for j, step in enumerate(sim_steps):
                tic = time.time()
                dt = Generator(self.config, integ=integ, step=step)
                dt.genImage()
                toc = time.time()
                print(f'Data for {integ} ms integ, {step} ns step generated in {toc-tic} seconds')
                
                tic = time.time()
                fit = Fitter(self.config, numsteps=dt.numsteps, step=step)
                results = fit.fit_exps(image=dt.image)
                
                self.means[0][i][j] += np.mean(results[2])
                self.stdevs[0][i][j] += np.std(results[2])
                if (len(self.config['lifetimes']) > 1):
                    self.means[1][i][j] += np.mean(results[3])
                    self.stdevs[1][i][j] += np.std(results[3])

                toc = time.time()
                print(f'Data analyzed in {toc-tic} seconds\n')

        Generator.plotLifetimes(self.means, self.stdevs, self.config['integ'], self.config['step'], self.config['lifetimes'], show=True)

    def run_single(self): # single vals (not in array) for 'integrations', 'gatesteps', and 'lifetimes' fields in .json
        tic = time.time()
        dt = Generator(self.config)
        dt.genImage()
        toc = time.time()
        print(f'Data generated in {toc-tic} seconds')

        print(dt.image)

        tic = time.time()
        fit = Fitter(self.config, numsteps=self.config['numsteps'], step=self.config['step'])
        results = fit.fit_exps(image=dt.image)
        fit.save_results(self.config['filename'], results)
        toc = time.time()
        print(f"Exponential fitting done in {toc-tic} seconds: {self.config['filename']}_fit_results.npz")
        
    def plot_single(self):
        plot = Plotter(self.config)
        results = np.load(self.config['filename'] + '_fit_results.npz')
        plot.plot_all(results, self.config['filename'], show=show)
        print(f"Results plotted: {self.config['filename']}_results.png")

if __name__ == '__main__':
    obj = Simulator(config_path)
    obj.run_full(sim_steps, sim_integs)