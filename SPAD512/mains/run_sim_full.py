import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread, imsave
import json
import time
from SPAD512.exps import Fitter, Plotter
from SPAD512.sims import Generator, plot_lifetimes, plot_fvals

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
        self.config_copy = self.config.copy()

        self.param1, self.param2 = self.config['test_params']
        self.means = np.zeros((len(self.config['lifetimes']), len(self.config[self.param1]), len(self.config[self.param2])))
        self.stdevs = np.zeros((len(self.config['lifetimes']), len(self.config[self.param1]), len(self.config[self.param2])))
        self.counts = np.zeros((len(self.config[self.param1]), len(self.config[self.param2])))
        self.f_vals = np.zeros((len(self.config[self.param1]), len(self.config[self.param2])))

    def post_sim(self, results, i, j):
        nonzero = results[2][(results[2] != 0) & (~np.isnan(results[2]))]
        self.means[0, i, j] += np.mean(nonzero)
        self.stdevs[0, i, j] += np.std(nonzero)
        self.counts[i,j] += np.mean(results[4])
        precs = self.stdevs[0,i,j]/self.means[0,i,j]
        accs = np.sqrt((self.means[0,i,j] - self.config['lifetimes'][0])**2)

        if len(self.config['lifetimes']) > 1:
            nonzero = results[3][(results[3] != 0) & (~np.isnan(results[3]))]
            self.means[1, i, j] += np.mean(nonzero)
            self.stdevs[1, i, j] += np.std(nonzero)
            precs += self.stdevs[1,i,j]/self.means[1,i,j]

            if self.means[0,i,j] < self.means[1,i,j]:
                temp = self.means[1,i,j]
                self.means[1,i,j] = self.means[0,i,j]
                self.means[0,i,j] = temp
                temp = self.stdevs[1,i,j]
                self.stdevs[1,i,j] = self.stdevs[0,i,j]
                self.stdevs[0,i,j] = temp

            if self.config['lifetimes'][0] < self.config['lifetimes'][1]:
                temp = self.config['lifetimes'][1]
                self.config['lifetimes'][1] = self.config['lifetimes'][0] 
                self.config['lifetimes'][0] = temp

            accs = np.sqrt((self.means[0,i,j] - self.config['lifetimes'][0])**2 + (self.means[1,i,j] - self.config['lifetimes'][1])**2)
        self.f_vals[i,j] = np.sqrt(self.counts[i,j]) * accs * precs

        self.means[np.isnan(self.means)] = 0 
        self.stdevs[np.isnan(self.stdevs)] = 100
        np.savez(self.config['filename'] + '_results.npz', means=self.means, stdevs=self.stdevs, counts=self.counts, f_vals = self.f_vals)

    def run_full(self):
        for i, curr_param1 in enumerate(self.config[self.param1]):
            for j, curr_param2 in enumerate(self.config[self.param2]):
                self.config_copy[self.param1] = curr_param1
                self.config_copy[self.param2] = curr_param2

                print(f'Generating data for {(curr_param1):.3f} {self.param1}, {(curr_param2):.3f} {self.param2}')
                tic = time.time()
                dt = Generator(self.config_copy)
                dt.genImage()
                toc = time.time()
                print(f'Done in {(toc-tic):.1f} seconds, now analyzing.')

                tic = time.time()
                fit = Fitter(self.config_copy, numsteps=dt.numsteps, times=dt.times)
                results = fit.fit_exps(image=dt.image)
                self.post_sim(results, i, j)
                toc = time.time()
                print(f'Data analyzed in {(toc-tic):.1f} seconds. F\'-value {self.f_vals[i,j]:.3f} \n')

    def plot_full(self,show=True):
        results = np.load(self.config['filename'] + '_results.npz')
        self.means = results['means'].astype(float)
        self.stdevs = results['stdevs'].astype(float)
        self.counts = results['counts'].astype(int)
        self.f_vals = results['f_vals'].astype(float)

        # plot_fvals(self.f_vals, self.config[self.param1], self.config[self.param2], self.config['filename'], show=show)
        plot_lifetimes(self.means, self.stdevs, self.config['bits'], self.config['integ'], self.config['lifetimes'], self.config['filename'], show=show)

if __name__ == '__main__':
    obj = Simulator(config_path)
    # obj.run_full()
    obj.plot_full()