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

config_path = 'SPAD512/SPAD512/mains/run_simulation.json'
show = True # show final plot

class Simulator:
    def __init__(self, config_path):
        with open(config_path) as f:
            self.config = json.load(f)

    def run_full(self): # array of vals for 'integrations', 'gatesteps', and 'lifetimes' fields in .json
        self.means = np.zeros((len(self.config['lifetimes']),len(self.config['integ']), len(self.config['step'])))
        self.stdevs = np.zeros((len(self.config['lifetimes']),len(self.config['integ']), len(self.config['step'])))

        for i, integ in enumerate(self.config['integ']):
            for j, step in enumerate(self.config['step']):
                tic = time.time()
                dt = Generator(self.config, integ=integ, step=step)
                dt.genImage()
                toc = time.time()
                print(f'Data for {(integ * 1e-3):.3f} ms integ, {(step * 1e-3):.3f} ns step generated in {(toc-tic):.1f} seconds')
                
                tic = time.time()
                fit = Fitter(self.config, numsteps=dt.numsteps, step=step)
                results = fit.fit_exps(image=dt.image)
                
                nonzero = results[2][results[2] != 0]
                self.means[0,i,j] += np.mean(nonzero)
                self.stdevs[0,i,j] += np.std(nonzero)

                if (len(self.config['lifetimes']) > 1):
                    nonzero = results[3][results[3] != 0]
                    self.means[1,i,j] += np.mean(nonzero)
                    self.stdevs[1,i,j] += np.std(nonzero)

                toc = time.time()
                print(f'Data analyzed in {(toc-tic):.1f} seconds. Mean lifetimes {(self.means[0,i,j]):.2f} ns\n')
        np.savez(self.config['filename'] + '_results.npz', means=self.means, stdevs=self.stdevs)

    def plot_sim(self,show=True):
        results = np.load(self.config['filename'] + '_results.npz')
        self.means = results['means'].astype(float)
        self.stdevs = results['stdevs'].astype(float)

        plotLifetimes(self.means, self.stdevs, self.orig_integs, self.orig_steps, self.config['lifetimes'], self.config['filename'] + '_results', show=show)



    def run_single(self): # single vals (not in array) for 'integrations', 'gatesteps', and 'lifetimes' fields in .json
        tic = time.time()
        dt = Generator(self.config, numsteps=self.config['numsteps'])
        dt.genImage()
        toc = time.time()
        print(f'Data generated in {(toc-tic):.1f} seconds')

        tic = time.time()
        fit = Fitter(self.config, numsteps=self.config['numsteps'], step=self.config['step'])
        results = fit.fit_exps(image=dt.image)
        fit.save_results(self.config['filename'], results)
        toc = time.time()
        print(f"Exponential fitting done in {(toc-tic):.1f} seconds: {self.config['filename']}_fit_results.npz")
        
    def plot_single(self,show=True):
        plot = Plotter(self.config)
        results = np.load(self.config['filename'] + '_fit_results.npz')
        plot.plot_all(results, self.config['filename'], show=show)
        print(f"Results plotted: {self.config['filename']}_results.png")

if __name__ == '__main__':
    obj = Simulator(config_path)
    obj.run_full()
    obj.plot_sim(show=show)