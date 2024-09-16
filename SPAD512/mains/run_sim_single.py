import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread, imsave
import json
import time
from SPAD512.exps import Fitter, Plotter
from SPAD512.sims import Generator

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

config_path = "C:\\Users\\ishaa\\Documents\\FLIM\\SPAD512\\SPAD512\\mains\\run_sim_single.json"
show = True # show final plot

class Simulator:
    def __init__(self, config_path):
        with open(config_path) as f:
            self.config = json.load(f)

    def gen(self):
        print('Generating data')
        tic = time.time()
        dt = Generator(self.config)
        dt.genImage()
        toc = time.time()
        print(f'Done in {(toc-tic):.1f} seconds. Analyzing')
        return dt

    def run(self, dt, subname=''): # single vals (not in array) for 'integrations', 'gatesteps', and 'lifetimes' fields in .json
        tic = time.time()
        fit = Fitter(self.config, numsteps=dt.numsteps, times=dt.times)
        results = fit.fit_exps(image=dt.image)
        fit.save_results(self.config['filename'] + subname, results)
        toc = time.time()
        print(f"{self.config['fit']} fitting done in {(toc-tic):.1f} seconds: {self.config['filename'] + '_nnls'}_fit_results.npz")
        return results
        
    def plot(self, subname='',show=True):
        results = np.load(self.config['filename'] + subname + '_fit_results.npz')
        plot = Plotter(self.config)
        # plot.plot_all(results, self.config['filename'] + subname, show=show) 

        A1, A2, tau1, tau2, intensity, full_trace, full_params, track, times = plot.preprocess_results(results)
        plot.plot_hist(tau1, tau2, splice = (8, 17), filename=self.config['filename'] + subname + '_lifetime_histogram.png', show=show)

        # print(f"Results plotted: {self.config['filename']}_results.png")

if __name__ == '__main__':
    obj = Simulator(config_path)
    # dt = obj.gen()
    # results = obj.run(dt)
    obj.plot()
    

