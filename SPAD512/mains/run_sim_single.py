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
        # print('Generating data')
        tic = time.time()
        dt = Generator(self.config)
        dt.genImage()
        toc = time.time()
        # print(f'Done in {(toc-tic):.1f} seconds. Analyzing')
        return dt

    def run(self, dt, subname=''): # single vals (not in array) for 'integrations', 'gatesteps', and 'lifetimes' fields in .json
        tic = time.time()
        fit = Fitter(self.config, numsteps=dt.numsteps, times=dt.times)
        results = fit.fit_exps(image=dt.image)
        fit.save_results(self.config['filename'] + subname, results)
        toc = time.time()
        # print(f"{self.config['fit']} fitting done in {(toc-tic):.1f} seconds: {self.config['filename'] + subname}_fit_results.npz")
        # print(f"Fitting done in {(toc-tic):.1f} seconds.")
        return results
        
    def plot(self, subname='',show=True):
        results = np.load(self.config['filename'] + subname + '_fit_results.npz')
        plot = Plotter(self.config)
        # plot.plot_all(results, self.config['filename'] + subname, show=show) 

        A1, A2, tau1, tau2, intensity, full_trace, full_params, track, times = plot.preprocess_results(results)
        plot.plot_hist(tau1, tau2, splice = (8, 17), filename=self.config['filename'] + subname + '_lifetime_histogram.png', show=show)

        print(f"Results plotted: {self.config['filename']}_results.png")

if __name__ == '__main__':
    iter = 100
    arr_bins = [50, 75, 100, 125, 150, 175, 200, 250, 300, 350, 400, 450, 500]
    # arr_bins = [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 22, 24, 26, 28]
    filename = 'c:\\Users\\ishaa\\Documents\\FLIM\\241202\\8bit_4step.npz'

    obj = Simulator(config_path)
    tau1s = np.zeros(len(arr_bins))
    tau2s = np.zeros(len(arr_bins))
    
    for i, bins in enumerate(arr_bins):
        tic = time.time()
        obj.config['x'] = bins
        obj.config['y'] = iter
        dt = obj.gen()
        
        dt.image[:, 0, :] = np.float64(np.sum(dt.image, axis=1))
        dt.image = np.float64(dt.image[:, :1, :])/bins
        obj.config['x'] = 1 #  need to relabel dimensions to take advantage of vectorization in analysis after rebinning

        results = obj.run(dt)        

        def spliced_std(data):
            data = np.ravel(data)

            q1 = np.percentile(data, 25)
            q3 = np.percentile(data, 75)
            iqr = q3-q1
            upper = q3 + 1.5*iqr
            lower = q1 - 1.5*iqr

            filtered = data[(data > lower) & (data < upper)]
            return np.std(filtered)/np.mean(filtered)


        tau1s[i] += spliced_std(results[2]) # results should already be sorted into higher and lower component
        tau2s[i] += spliced_std(results[3])

        toc = time.time()
        print(f'{arr_bins[i]} bins done in {(toc-tic):.1f} s: {tau1s[i]:.3f}, {tau2s[i]:.3f} \n \n')
    print(f'Overall precision: {tau1s}, {tau2s} \n \n')

    with open(config_path, 'r') as f:
        metadt = json.load(f)

    np.savez(filename, 
             tau1s=tau1s,
             tau2s=tau2s, 
             iter=iter, 
             arr_bins=arr_bins, 
             metadata=metadt)
    
    plt.plot(arr_bins, tau1s, 'o-b', label='20 ns true')
    plt.plot(arr_bins, tau2s, 'o-k', label='5 ns true')
    plt.grid(True)
    plt.legend()
    plt.title('4-bit RSD improvement with binning')
    plt.xlabel('Number of binned pixels')
    plt.ylabel('Relative standard deviation')
    plt.show()