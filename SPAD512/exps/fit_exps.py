import numpy as np
import scipy.optimize as opt
import emcee
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
    
    def decay_double(self, x, amp1, tau1, amp2, tau2):
        return amp1 * np.exp(-x / tau1) + amp2 * np.exp(-x / tau2)

    def fit_decay(self, times, data):
        if (self.config['components'] == 1):
            initial_guess = [np.max(data), 2.0]
            params, _ = opt.curve_fit(self.decay, times, data, p0=initial_guess)
            return params
        elif (self.config['components'] == 2):
            initial_guess = [np.max(data), 2.0, np.max(data)/2, 1.0]
            params, _ = opt.curve_fit(self.decay_double, times, data, p0=initial_guess)
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
                    self.intensity[i][j] += np.sum(trace)

                    loc = np.argmax(trace)
                    try:
                        params = self.fit_decay(self.times[loc:], trace[loc:])
                        self.track += 1
                    except RuntimeError:
                        params = [0, 0]

                    if (self.config['components'] == 2):
                        self.A[i][j] += (params[0], params[2])
                        self.tau[i][j] += (params[1], params[3])

        loc = np.argmax(self.full_trace)
        try:
            params = self.fit_decay(self.times[loc:], self.full_trace[loc:])
        except RuntimeError:
            params = [0, 0]

        return self.A, self.intensity, self.tau, self.times, self.full_trace, params, self.track
    
    # Bayesian method for getting double exponential info (incomplete, need to adjust params return)
    def like(self, params, times, data):
        model = self.decay_double(times, params[0], params[1], params[2], params[3])
        sigma2 = 0.1**2
        return -0.5 * np.sum((data - model)**2 / sigma2 + np.log(sigma2))

    def prior(self, params):
        if any(param <= 0 for param in params):
            return -np.inf
        return 0

    def prob(self, params, times, data):
        lp = self.prior(params)
        if not np.isfinite(lp):
            return -np.inf
        return lp + self.like(params, times, data)

    def mcmc(self, times, data):
        ndim = 4  # can adjust this if triple exponential is desired
        walkers = 32
        guess = [np.max(data), 2.0, np.max(data) / 2, 1.0] 
        pos = guess + (1e-4 * np.random.randn(walkers, ndim))

        sampler = emcee.EnsembleSampler(walkers, ndim, self.prob, args=(times, data))
        sampler.run_mcmc(pos, 5000, progress=True)

        samples = sampler.get_chain(discard=100, thin=15, flat=True)
        params = np.mean(samples, axis=0)
        return params

    def save_results(self, filename, results):
        np.savez(filename + '_fit_results.npz', A=results[0], intensity=results[1], tau=results[2], times=results[3], full_trace=results[4], params=results[5], track=results[6])