import numpy as np
from scipy import optimize as opt
from skimage.io import imread
from concurrent.futures import ProcessPoolExecutor, as_completed

class Fitter:
    def __init__(self, config, numsteps=0, step=0):
        self.config = config

        if numsteps:
            self.times = (np.arange(numsteps) * step) + config['offset']
            self.numsteps = numsteps
        else:
            self.times = (np.arange(config['numsteps']) * config['step']) + config['offset']
            self.numsteps = config['numsteps']

        self.A = None
        self.intensity = None
        self.tau = None
        self.full_trace = None
        self.track = 0

    def decay(self, x, amp, tau):
        return amp * np.exp(-x / tau)
    
    def decay_double(self, x, amp1, tau1, amp2, tau2):
        return amp1 * np.exp(-x / tau1) + amp2 * np.exp(-x / tau2)

    def fit_decay(self, times, data):
        if self.config['components'] == 1:
            initial_guess = [np.max(data), 2.0]
            params, _ = opt.curve_fit(self.decay, times, data, p0=initial_guess)
            return params

        elif self.config['components'] == 2:
            initial_guess = [np.max(data), 2.0, np.max(data) / 2, 1.0]
            params, _ = opt.curve_fit(self.decay_double, times, data, p0=initial_guess)
            return params

    def fit_trace(self, trace, i, j):
        passed = False
        if np.sum(trace) > self.config['thresh']:
            passed = True
            self.intensity[i][j] += np.sum(trace)
            loc = np.argmax(trace)

            if self.config['components'] == 1:
                try:
                    params = self.fit_decay(self.times[loc:], trace[loc:])
                    self.track += 1
                except RuntimeError:
                    params = [0, 0, 0 , 0]
                return (params[0], 0, params[1], 0, i, j, passed)
            
            elif self.config['components'] == 2:
                try:
                    params = self.fit_decay(self.times[loc:], trace[loc:])
                    self.track += 1
                except RuntimeError:
                    params = [0, 0, 0, 0]
                return (params[0], params[2], params[1], params[3], i, j, passed)
            
            return (0, 0, 0, 0, i, j, passed)
        
        else:
            trace = np.zeros((self.numsteps), dtype=float)

    def fit_exps(self, filename=None, image=None):
        if filename:
            image = imread(filename + '.tif')
            length, x, y = np.shape(image)
        elif image is not None:
            length, x, y = np.shape(image)
        else:
            print('no filename or image given')
            return 0

        self.A = np.zeros((x, y), dtype=float)
        self.intensity = np.zeros((x, y), dtype=float)
        self.tau = np.zeros((x, y), dtype=float)
        self.full_trace = np.zeros((self.numsteps), dtype=float)

        with ProcessPoolExecutor() as executor:
            futures = [executor.submit(self.fit_trace, image[:self.numsteps, i, j], i, j) for i in range(x) for j in range(y)]
            for future in as_completed(futures):
                amp1, amp2, tau1, tau2, i, j, passed = future.result() # need to add processing code for amp2 and tau2 multi-exponent fit
                if passed:
                    self.full_trace += image[:self.numsteps, i, j]
                self.A[i][j] += amp1
                self.tau[i][j] += tau1

        loc = np.argmax(self.full_trace)
        try:
            params = self.fit_decay(self.times[loc:], self.full_trace[loc:])
        except RuntimeError:
            params = [0, 0]

        return self.A, self.intensity, self.tau, self.times, self.full_trace, params, self.track
    
    def save_results(self, filename, results):
        np.savez(filename + '_fit_results.npz', A=results[0], intensity=results[1], tau=results[2], times=results[3], full_trace=results[4], params=results[5], track=results[6])   