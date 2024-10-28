import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread, imsave
import json
import time
from SPAD512.bayes import BNP, Plotter

config_path = 'SPAD512/SPAD512/mains/run_bnps.json'
show = True # show final plot

class Analyzer:
    def __init__(self, config_path):
        with open(config_path) as f:
            self.config = json.load(f)

    def run(self, show=True):      
        tic = time.time()

        dp = BNP(self.config)
        dp.image_BNP()
        dp.save_results()

        toc = time.time()
        print(f"Full sampling done in {toc-tic} seconds: {self.config['filename']}_bnp_results.npz")


info = Analyzer(config_path)
info.run(show=show)