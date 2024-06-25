import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread, imsave
import json
import time
from SPAD512.exps import Fitter, Plotter
from SPAD512.utils import GatedReader

read = False # read from folder (if read=True, file var below is irrelevant)
file = '240613/240613_SPAD-QD-10MHz-3f-5000g-1000us-5ns-18ps-18ps-150uW' # make sure to remove .tif suffix
config_path = 'SPAD512/SPAD512/mains/run_exponential.json' # no leading slash, keep .json suffix
show = True # show final plot

class Analyzer:
    def __init__(self, config_path):
        with open(config_path) as f:
            self.config = json.load(f)

    def run(self, show=True, read=False, file=None):
        if (read==True):
            data = GatedReader(self.config)
            filename = data.create_data()
            print(f"Data created: {filename}")
        else:
            filename = file
        
        # tic = time.time()
        # fit = Fitter(self.config)
        # results = fit.fit_exps(filename=filename)
        # fit.save_results(filename, results)
        # toc = time.time()
        # print(f"Exponential fitting done in {toc-tic} seconds: {filename}_fit_results.npz")

        plot = Plotter(self.config)
        results = np.load(filename + '_fit_results.npz')
        plot.plot_all(results, filename, show=show)
        print(f"Results plotted: {filename}_results.png")

if __name__=='__main__':
    info = Analyzer(config_path)
    info.run(show=show, read=read, file=file)

# utils/read from folder
# option for utils/rebin
# exps/thresh
# exps/fit
# exps/plot