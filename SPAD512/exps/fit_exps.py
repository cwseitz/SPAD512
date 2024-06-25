import numpy as np
from scipy import optimize as opt
from skimage.io import imread
from concurrent.futures import ProcessPoolExecutor, as_completed
from scipy.signal import convolve, deconvolve
from scipy.special import erfc

class Fitter:
    def __init__(self, config, numsteps=0, step=0):
        self.config = config

        if numsteps:
            self.times = (np.arange(numsteps) * step) + config['offset']
            self.numsteps = numsteps
            self.step = step
        else:
            self.times = (np.arange(config['numsteps']) * config['step']) + config['offset']
            self.numsteps = config['numsteps']
            self.step = config['step']

        self.A1 = None
        self.tau1 = None
        self.A2 = None
        self.tau2 = None
        self.intensity = None
        self.full_trace = None
        self.track = 0    

    def decay(self, x, amp, tau):
        return amp * np.exp(-x / tau)
    
    def decay_double(self, x, amp1, tau1, amp2, tau2):
        return amp1 * np.exp(-x / tau1) + amp2 * np.exp(-x / tau2)

    def decay_conv(self, x, A, lam):
        sigma = self.config['irf_width']/self.step
        
        term1 = (1/2) * A * np.exp((1/2) * lam * (2*self.config['irf_mean'] + lam*(sigma**2) - 2*x))
        term2 = lam * erfc((self.config['irf_mean'] + lam*(sigma**2) - x)/(sigma*np.sqrt(2)))
        
        return term1*term2

    # def decay_double_conv(self, x, A, lam):
    #     sigma = self.config['irf_width']/self.step
        
    #     term1 = (1/2) * A * np.exp((1/2) * lam * (2*self.config['irf_mean'] + lam*(sigma**2) - 2*x))
    #     term2 = lam * erfc((self.config['irf_mean'] + lam*(sigma**2) - x)/(sigma*np.sqrt(2)))
        
    #     return term1*term2


    def fit_decay(self, times, data):
        if self.config['components'] == 1:
            initial_guess = [np.max(data), 2.0]
            params, _ = opt.curve_fit(self.decay_conv, times, data, p0=initial_guess)
            return params

        elif self.config['components'] == 2:
            initial_guess = [np.max(data), 2.0, np.max(data) / 2, 1.0]
            params, _ = opt.curve_fit(self.decay_double, times, data, p0=initial_guess)
            return params

    def fit_trace(self, trace, i, j):
        success = False
        if np.sum(trace) > self.config['thresh']:
            loc = np.argmax(trace)

            if self.config['components'] == 1:
                try:
                    params = self.fit_decay(self.times[loc:], trace[loc:])
                    params[1] = 1/params[1]
                    success = True
                except RuntimeError:
                    params = [0, 0, 0 , 0]
                return (params[0], params[1], 0, 0, i, j, success)
            
            elif self.config['components'] == 2:
                try:
                    params = self.fit_decay(self.times[loc:], trace[loc:])
                    params[1] = 1/params[1]
                    params[3] = 1/params[3]
                    success = True
                except RuntimeError:
                    params = [0, 0, 0, 0]
                return (params[0], params[1], params[2], params[3], i, j, success)

        return (0, 0, 0, 0, i, j, success)

    def fit_exps(self, filename=None, image=None):
        if filename:
            image = imread(filename + '.tif')
            length, x, y = np.shape(image)
        elif image is not None:
            length, x, y = np.shape(image)
        else:
            print('no filename or image given')
            return 0

        self.A1 = np.zeros((x, y), dtype=float)
        self.A2 = np.zeros((x, y), dtype=float)
        self.tau1 = np.zeros((x, y), dtype=float)
        self.tau2 = np.zeros((x, y), dtype=float)
        self.intensity = np.zeros((x, y), dtype=float)
        self.full_trace = np.zeros((self.numsteps), dtype=float)

        with ProcessPoolExecutor() as executor:
            print('entering parallelization')
            futures = [executor.submit(self.fit_trace, image[:self.numsteps, i, j], i, j) for i in range(x) for j in range(y)]
            for future in as_completed(futures):
                amp1, tau1, amp2, tau2, i, j, success = future.result()
                if success:
                    self.A1[i][j] += amp1
                    self.A2[i][j] += amp2
                    self.tau1[i][j] += tau1
                    self.tau2[i][j] += tau2
                    self.full_trace += image[:self.numsteps, i, j]
                    self.intensity[i][j] += np.sum(image[:self.numsteps, i, j])
                    self.track += 1
                    print(f'Pixel ({i}, {j}) fit: {tau1}, {tau2}')

        try:
            params = self.fit_trace[self.full_trace, 0, 0]
        except RuntimeError:
            params = [0, 0]

        print(params)

        return self.A1, self.A2, self.tau1, self.tau2, self.intensity, self.full_trace, params, self.track, self.times, 
    
    def save_results(self, filename, results):
        np.savez(
            filename + '_fit_results.npz', 
            A1=results[0], 
            A2=results[1], 
            tau1=results[2], 
            tau2=results[3], 
            intensity=results[4], 
            full_trace=results[5], 
            full_params=results[6], 
            track=results[7],
            times=results[8]
        )   