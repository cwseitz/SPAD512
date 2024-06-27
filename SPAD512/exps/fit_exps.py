import numpy as np
from scipy import optimize as opt
import tifffile as tf
from concurrent.futures import ProcessPoolExecutor, as_completed
from scipy.signal import convolve, deconvolve
from scipy.special import erfc

class Trace:
    def __init__(self, config, data, i, j):
        self.config = config
        self.step = config['step']
        self.times = config['times']
        self.curve = config['curve']  # Options for curve: 'mono', 'mono_conv', 'bi', 'bi_conv'
        self.irf_width = config['irf_width']
        self.irf_mean = config['irf_mean']
        self.thresh = config['thresh']
        del self.config

        self.data = data
        self.i = i
        self.j = j
        
        self.success = False
        self.sum = np.sum(self.data)
    


    '''Helper methods for LMA fitting'''
    def mono(self, x, amp, tau):
        return amp * np.exp(-x / tau)
    
    def bi(self, x, amp1, tau1, amp2, tau2):
        return amp1 * np.exp(-x / tau1) + amp2 * np.exp(-x / tau2)

    def mono_conv(self, x, A, lam):
        sigma = self.irf_width/self.step

        term1 = (1/2) * A * np.exp((1/2) * lam * (2*float(self.irf_mean) + lam*(sigma**2) - 2*x))
        term2 = lam * erfc((float(self.irf_mean) + lam*(sigma**2) - x)/(sigma*np.sqrt(2)))

        return term1*term2

    def log_mono_conv(self, x, A, lam):
        sigma = self.irf_width / self.step

        log_term1 = np.log(A) + (1/2) * lam * (2*self.irf_mean + lam*(sigma**2) - 2*x)
        log_term2 = np.log(lam) + np.log(erfc((self.irf_mean + lam*(sigma**2) - x) / (sigma * np.sqrt(2))))

        full = np.exp(log_term1 + log_term2)

        return full

    # def bi_conv(self, x, A, lam):
    #     sigma = self.config['irf_width']/self.step
        
    #     term1 = (1/2) * A * np.exp((1/2) * lam * (2*self.config['irf_mean'] + lam*(sigma**2) - 2*x))
    #     term2 = lam * erfc((self.config['irf_mean'] + lam*(sigma**2) - x)/(sigma*np.sqrt(2)))
        
    #     return term1*term2



    '''Helper methods for Metropolis-Hastings'''
    def log_like(self, params, x, y, func):
        A, lam = params
        ym = func(x, A, lam)
        return -0.5 * np.sum((y - ym)**2)

    def proposal(self, params):
        return params + np.random.normal(0, 0.1, size=params.shape)

    def met_hast(self, x, y, func, guess_params, iter=10000):
        curr_params = guess_params
        curr_like = self.log_like(curr_params, x, y, func)
        samples = []
        
        for _ in range(iter):
            prop_params = self.proposal(curr_params)
            prop_like = self.log_like(prop_params, x, y, func)
            
            if np.log(np.random.rand()) < (prop_like - curr_like):
                curr_params = prop_params
                curr_like = prop_like
                
            samples.append(curr_params)
        
        return np.array(samples)



    def fit_decay(self):
        match self.curve:
            case 'mono':
                loc = np.argmax(self.data)
                guess = [np.max(self.data), 2.0]
                params, _ = opt.curve_fit(self.mono, self.times[loc:], self.data[loc:], p0=guess)
                return (params[0], params[1], 0, 0) 

            case 'bi':
                loc = np.argmax(self.data)
                guess = [np.max(self.data), 2.0, np.max(self.data) / 2, 1.0]
                params, _ = opt.curve_fit(self.bi, self.times[loc:], self.data[loc:], p0=guess)
                return params 

            case 'mono_conv':
                loc = np.argmax(self.data)
                guess = [np.max(self.data), 2]
                params, _ = opt.curve_fit(self.mono_conv, self.times[loc:], self.data[loc:], p0=guess)
                return (params[0], params[1], 0, 0) 

            case 'log_mono_conv':
                loc = np.argmax(self.data)
                guess = [np.max(self.data), 2.0]
                params, _ = opt.curve_fit(self.log_mono_conv, self.times[loc:], self.data[loc:], p0=guess)
                return (params[0], params[1], 0, 0)
            
            case 'mono_conv_mcmc':
                guess = np.array([1.0, 0.5])
                samples = self.met_hast(self.times, self.data, self.log_mono_conv, guess)
                return (np.mean(samples[3000:,0]), np.mean(samples[3000:,1]), 0 ,0)

            case 'bi_conv':
                return 0

            case _:
                raise Exception('Curve choice invalid in config.json, choose from mono, bi, mono_conv, and bi_conv.')

    def fit_trace(self):
        if self.sum > self.thresh:
            try:
                self.params = self.fit_decay()
                self.success = True
            except RuntimeError:
                self.params = [0, 0, 0, 0]
        else: self.params = [0, 0, 0, 0]
    
class Fitter:
    def __init__(self, config, numsteps=0, step=0):
        self.config = config

        if numsteps:
            self.config['times'] = (np.arange(numsteps) * step) + config['offset']
            self.config['numsteps'] = numsteps
            self.config['step'] = step
        else:
            self.config['times'] = (np.arange(config['numsteps']) * config['step']) + config['offset']
            self.config['numsteps'] = numsteps = config['numsteps']
            self.config['step'] = config['step']

        self.A1 = None
        self.tau1 = None
        self.A2 = None
        self.tau2 = None
        self.intensity = None
        self.full_trace = None
        self.track = 0    

    @staticmethod
    def helper(config, data, i, j):
        dt = Trace(config, data, i, j)
        dt.fit_trace()
        return dt.params, dt.success, dt.sum, i, j


    def fit_exps(self, filename=None, image=None):
        if filename:
            with tf.TiffFile(filename + '.tif') as tif:
                image = tif.asarray(key=range(self.config['numsteps']))  # Only read the first 5000 frames
            length, x, y = np.shape(image)
        elif image is not None:
            image = image[:self.config['numsteps'],:,:]
            length, x, y = np.shape(image)
        else:
            raise Exception('No filename or image provided to fit_exps, make sure to provide one or the other.')

        self.A1 = np.zeros((x, y), dtype=float)
        self.A2 = np.zeros((x, y), dtype=float)
        self.tau1 = np.zeros((x, y), dtype=float)
        self.tau2 = np.zeros((x, y), dtype=float)
        self.intensity = np.zeros((x, y), dtype=float)
        self.full_trace = np.zeros((self.config['numsteps']), dtype=float)

        with ProcessPoolExecutor() as executor:
            futures = [executor.submit(self.helper, self.config, image[:, i, j], i, j) for i in range(x) for j in range(y)]
            for future in as_completed(futures):
                outputs, success, sum, i, j = future.result()
                if success:
                    self.A1[i][j] += outputs[0]
                    self.tau1[i][j] += 1/(outputs[1]+1e-10)
                    self.A2[i][j] += outputs[2]
                    self.tau2[i][j] += 1/(outputs[3]+1e-10)
                    self.intensity[i][j] += sum

                    self.full_trace += image[:, i, j]
                    self.track += 1
                    print(f'Pixel ({i}, {j}) fit. Parameters: {outputs}.')

        outputs, success, sum, i, j = self.helper(self.config, self.full_trace, 0, 0)

        return self.A1, self.A2, self.tau1, self.tau2, self.intensity, self.full_trace, outputs, self.track, self.config['times']
    
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