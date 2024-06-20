import numpy as np
import scipy.optimize as opt
from skimage.io import imread
import json

class Fitter:
    def __init__(self, config):
        self.config = config
        self.times = (np.arange(config['gate_num']) * config['gate_step']) + config['gate_offset']
        self.A = None
        self.intensity = None
        self.tau = None
        self.full_trace = None
        self.track = 0

    def decay(self, x, amp, tau):
        return amp * np.exp(-x / tau)

    def fit_decay(self, times, data):
        initial_guess = [np.max(data), 2.0]
        params, _ = opt.curve_fit(self.decay, times, data, p0=initial_guess)
        return params

    def fit_exps(self, filename):
        image = imread(filename + '.tif')
        length, x, y = np.shape(image)

        self.A = np.zeros((x, y), dtype=float)
        self.intensity = np.zeros((x, y), dtype=float)
        self.tau = np.zeros((x, y), dtype=float)
        self.full_trace = np.zeros((self.config['gate_num']), dtype=float)

        for i in range(x):
            for j in range(y):
                trace = image[:self.config['gate_num'], i, j]
                if (np.sum(trace) > self.config['thresh']):
                    self.full_trace += trace
                    self.intensity[i][j] += sum(trace)

                    loc = np.argmax(trace)
                    try:
                        params = self.fit_decay(self.times[loc:], trace[loc:])
                        self.track += 1
                    except RuntimeError:
                        params = [0, 0]

                    self.A[i][j] += params[0]
                    self.tau[i][j] += params[1]

        loc = np.argmax(self.full_trace)
        try:
            params = self.fit_decay(self.times[loc:], self.full_trace[loc:])
        except RuntimeError:
            params = [0, 0]

        return self.A, self.intensity, self.tau, self.times, self.full_trace, params, self.track

    def save_results(self, filename, results):
        np.savez(filename + '_fit_results.npz', A=results[0], intensity=results[1], tau=results[2], times=results[3], full_trace=results[4], params=results[5], track=results[6])