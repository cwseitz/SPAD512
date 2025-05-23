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
        # dt.plotTrace(show_max=True, correct=False, save=False, filename=r'C:\Users\ishaa\Documents\FLIM\figure_remaking\figure3_7o5ms-trace')
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
        # print(f"{self.config['fit']} fitting done in {(toc-tic):.1f} seconds: {self.config['filename'] + subname}_fit_results.npz")
        # print(f"Fitting done in {(toc-tic):.1f} seconds.")
        return results
        
    def plot(self, subname='',show=True):
        results = np.load(self.config['filename'] + subname + '_fit_results.npz')
        plot = Plotter(self.config)
        # plot.plot_all(results, self.config['filename'] + subname, show=show) 

        A1, A2, tau1, tau2, intensity, full_trace, full_params, track, times = plot.preprocess_results(results)
        plot.plot_hist(tau1, tau2, splice = (8, 17), filename=self.config['filename'] + subname + '_lifetime_histogram.png', show=show)
        # np.savez(r'C:\Users\ishaa\Documents\FLIM\figure_remaking\figure3_7o5ms-hist',
        #          tau1=tau1,
        #          tau2=tau2
        #         )
        # plot.plot_hist_unspliced(tau1, tau2, filename=self.config['filename'] + subname + '_lifetime_histogram.png', show=show)

        print(f"Results plotted: {self.config['filename']}_results.png")

    def run_bintest():
        iter = 100
        arr_bins = np.arange(1, 41)
        print(arr_bins)
        filename = 'c:\\Users\\ishaa\\Documents\\FLIM\\new_bintest\\12bit_25ms.npz'

        with open(config_path, 'r') as f:
            metadt = json.load(f)

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

            np.savez(filename, 
                tau1s=tau1s,
                tau2s=tau2s, 
                iter=iter, 
                arr_bins=arr_bins, 
                metadata=metadt)
            toc = time.time() 
            print(f'{arr_bins[i]} bins done in {(toc-tic):.1f} s: {tau1s[i]:.3f}, {tau2s[i]:.3f} \n \n')

        print(f'Overall precision: {tau1s}, {tau2s} \n \n')

        

        

    def run_integs():
        obj = Simulator(config_path)
        integs = [100, 250, 500, 1000, 2500, 5000, 7500, 10000, 12500, 15000, 17500, 20000, 22500, 25000]
        tau1s = np.zeros(len(integs))
        tau1s_err = np.zeros(len(integs))
        tau2s = np.zeros(len(integs))
        tau2s_err = np.zeros(len(integs))

        k = obj.config['kernel_size']

        for i, integ in enumerate(integs):
            obj.config['integ'] = integ
            dt = obj.gen()
            results = obj.run(dt)

            tau1s[i] += np.mean(results[2][k:-k, k:-k])
            tau1s_err[i] += np.std(results[2][k:-k, k:-k])
            tau2s[i] += np.mean(results[3][k:-k, k:-k])
            tau2s_err[i] += np.std(results[3][k:-k, k:-k])

            print(f'integ: {integ}, tau1: {tau1s[i]}, tau2: {tau2s[i]}\n')
        
        integs = np.array(integs)/1000

        np.savez(r'C:\Users\ishaa\Documents\FLIM\figure_remaking\figure3_timeseries',
                 integs=integs,
                 tau1s=tau1s,
                 tau1s_err=tau1s_err,
                 tau2s=tau2s,
                 tau2s_err=tau2s_err)
        
        plt.errorbar(integs, tau1s, yerr=tau1s_err, capsize=5, fmt='og', label='Longer lifetimes') 
        plt.errorbar(integs, tau2s, yerr=tau2s_err, capsize=5, fmt='ob', label='Shorter lifetimes')
        plt.ylim(0, 25)
        plt.axhline(20, color='green', linestyle='--', label='20 ns ground truth')
        plt.axhline(5, color='blue', linestyle='--', label = '5 ns ground truth')
       
        plt.xlabel('Integration time (ms)', fontsize=14)
        plt.ylabel('Lifetime, (ns)', fontsize=14)
        plt.xticks([0, 5, 10, 15, 20, 25])
        plt.yticks()
        plt.tick_params(axis='both', labelsize=12)

        plt.legend(fontsize=12)
        plt.show() 

    def run_json():
        obj = Simulator(config_path)
        dt = obj.gen()
        results = obj.run(dt)
        obj.plot()

if __name__ == '__main__':
    Simulator.run_json()