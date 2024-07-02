import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread, imsave
import json
import time
from SPAD512.exps import Fitter, Plotter
from SPAD512.utils import GatedReader

'''
Script for exponential analyses of time-gated FLIM data. .json formatting info below.

    "freq": MHz, pulsed laser frequency
    "frames": counts, number of decay profiles acquired
    "numsteps": counts, number of gate step increments
    "integ": us, exposure time for a single gate
    "width": ns, width of a single gate
    "step": ps, increment size between gates
    "offset": ps, initial increment for first gate step"freq": 10,
    "power": uW, laser power
    "bits": counts, bit depth of acquisition
    "globstrs_1bit": [], 1-bit image info
    "folder": str, name and path of folder holding images to stack
    "roi_dim": count, size of square to extract as ROI from top left
        Note: All entries power --> roi_dim unneeded if not reading from .tif
    "thresh": counts, sum of trace over pixel needed to not discard pixel in analysis
    "irf_mean": ns, mean value of gaussian IRF
    "irf_width": ns, stdev of gaussian IRF
    "fit": ('mono', 'mono_conv', 'log_mono_conv', 'mh_mono_conv', 'bi'), analysis method to use
'''

read = False # read from folder (if read=False {default}, must specify a file below)
file = '240613_SPAD-QD-10MHz-3f-900g-1000us-5ns-100ps-18ps-150uW' # make sure to remove .tif suffix
config_path = 'mains\\run_exponential.json' # no leading slash, keep .json suffix
show = True # show final plot

class Analyzer:
    def __init__(self, config_path, read=False, file=None):
        with open(config_path) as f:
            self.config = json.load(f)
        if read:
            data = GatedReader(self.config)
            self.filename = data.create_data()
            print(f"Data created: {self.filename}")
        else:
            self.filename = file

    def fit(self):
        tic = time.time()
        fit = Fitter(self.config)
        results = fit.fit_exps(filename=self.filename)
        fit.save_results(self.filename, results)
        toc = time.time()
        print(f"Exponential fitting done in {toc-tic} seconds: {self.filename}_fit_results.npz")

    def plot(self, show=True):
        plot = Plotter(self.config)
        results = np.load(self.filename + '_fit_results.npz')
        plot.plot_all(results, self.filename, show=show)
        print(f"Results plotted: {self.filename}_results.png")

if __name__=='__main__':
    info = Analyzer(config_path, file=file)
    info.fit()
    info.plot(show=show)