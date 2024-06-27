import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread, imsave
import json
import time

from SPAD512.exps import Fitter, Plotter
from SPAD512.utils import Generator

config_path = 'run_simulation.json'
show = True # show final plot

class Simulator:
    def __init__(self, config_path):
        with open(config_path) as f:
            self.config = json.load(f)
    
    def run_full(self): # array of vals for 'integrations', 'gatesteps', and 'lifetimes' fields in .json
        for tau in self.config['lifetimes']:
            self.means = np.zeros((len(self.config['integ']), len(self.config['step'])))
            self.stdevs = np.zeros((len(self.config['integ']), len(self.config['step'])))

            for i, integ in enumerate(self.config['integ']):
                for j, step in enumerate(self.config['step']):
                    tic = time.time()
                    dt = Generator(self.config, integ=integ, step=step, tau=tau)
                    dt.genImage()
                    toc = time.time()
                    print(f'Data for {tau} ns tau, {integ} ms integ, {step} ns step generated in {toc-tic} seconds')
                    
                    tic = time.time()
                    fit = Fitter(self.config, numsteps=dt.numsteps, step=step)
                    results = fit.fit_exps(image=dt.image)
                    self.means[i][j] += np.mean(results[2])
                    self.stdevs[i][j] += np.std(results[2])
                    toc = time.time()
                    print(f'Data analyzed in {toc-tic} seconds\n')

            Generator.plotLifetimes(self.means, self.stdevs, self.config['integ'], self.config['step'], tau, show=True)

    def run_single(self): # single vals (not in array) for 'integrations', 'gatesteps', and 'lifetimes' fields in .json
        tic = time.time()
        dt = Generator(self.config)
        dt.genImage()
        toc = time.time()
        print(f'Data generated in {toc-tic} seconds')

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
    obj.run_single()
    obj.plot_single()