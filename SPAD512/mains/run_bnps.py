import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread, imsave
import json
import time
from SPAD512.bnps import BNP
from SPAD512.utils import Reader

config_path = 'SPAD512/mains/run_exponential.json'
show = True # show final plot

class Analyzer:
    def __init__(self, config_path):
        with open(config_path) as f:
            self.config = json.load(f)

    def run(self):      
        tic = time.time()
        fit = BNP(self.config)
        results = fit.fit_exps(self.config['filename'])
        fit.save_results(self.config['filename'], results)
        toc = time.time()
        print(f"Full sampling done in {toc-tic} seconds: {self.config['filename']}_fit_results.npz")


info = Analyzer(config_path)
info.run(show=show)