import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread, imsave
import json
import time
from SPAD512.exps import Fitter, Plotter
from SPAD512.utils import Reader

read = False # read from folder (data not stacked into .tif yet)
file = '240613/240613_SPAD-QD-10MHz-3f-5000g-1000us-5ns-18ps-18ps-150uW' # irrelevant if read = True
config_path = 'SPAD512/mains/run_exponential.json'
show = False # show final plot

class Analyzer:
    def __init__(self, config_path):
        with open(config_path) as f:
            self.config = json.load(f)

    def run(self, show=True, read=False, file = ''):
        if (read==True):
            data = Reader(self.config)
            filename = data.create_data()
            print(f"Data created: {filename}")
        else:
            filename = file
        
        tic = time.time()
        filename = '240613/240613_SPAD-QD-10MHz-3f-5000g-1000us-5ns-18ps-18ps-150uW'
        fit = Fitter(self.config)
        results = fit.fit_exps(filename)
        fit.save_results(filename, results)
        toc = time.time()
        print(f"Exponential fitting done in {toc-tic} seconds: {filename}_fit_results.npz")

        plot = Plotter(self.config)
        results = np.load(filename + '_fit_results.npz')
        plot.plot_all(results, filename, show=show)
        print(f"Results plotted: {filename}_results.png")


info = Analyzer(config_path)
info.run(show=show, read=read, file=file)

# utils/read from folder
# option for utils/rebin
# exps/thresh
# exps/fit
# exps/plot