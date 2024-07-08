import numpy as np
from scipy import optimize as opt
import tifffile as tf
from concurrent.futures import ProcessPoolExecutor, as_completed
from scipy.signal import convolve, deconvolve, butter, filtfilt
from scipy.fft import fft, ifft, fftfreq
from scipy.special import erfc
import matplotlib.pyplot as plt
import time

class Trace:
    def __init__(self, config, data, i, j):
        self.config = config
        self.step = config['step']
        self.times = config['times']
        self.curve = config['fit']  # Options for curve: 'mono', 'mono_conv', 'log_mono_conv', 'mh_mono_conv', 'bi'
        self.irf_width = config['irf_width']
        self.irf_mean = config['irf_mean']
        self.thresh = config['thresh']
        del self.config

        self.data = data
        self.i = i
        self.j = j
        
        self.success = False
        self.sum = np.sum(self.data)
    


    '''Fitting functions'''
    def mono(self, x, p):
        A, lam = p
        return A * np.exp(-lam * x)

    def mono_conv(self, x, p):
        A, lam = p
        term1 = (A*lam/2) * np.exp((1/2) * lam * (2*float(self.irf_mean) + lam*(self.irf_width**2) - 2*x))
        term2 = erfc((float(self.irf_mean) + lam*(self.irf_width**2) - x)/(self.irf_width*np.sqrt(2)))
        return term1*term2

    def log_mono_conv(self, x, p):
        A, lam = p
        if (A <= 0):
            A = 1e-10
        if (lam <= 0):
            lam = 1e-10
        
        term1 = np.log(A*lam/2) + ((1/2) * lam * (2*float(self.irf_mean) + lam*(self.irf_width**2) - 2*x))
        term2 = np.log(erfc((float(self.irf_mean) + lam*(self.irf_width**2) - x)/(self.irf_width*np.sqrt(2))))
        return term1 + term2
        
    def bi(self, x, p):
        A1, lam1, A2, lam2 = p
        return A1 * np.exp(-x * lam1) + A2 * np.exp(-x * lam2)

    def bi_conv(self, x, p):
        A1, lam1, A2, lam2 = p

        term1 = (A1*lam1/2) * np.exp((1/2) * lam1 * (2*float(self.irf_mean) + lam1*(self.irf_width**2) - 2*x))
        term2 = erfc((float(self.irf_mean) + lam1*(self.irf_width**2) - x)/(self.irf_width*np.sqrt(2)))
        term3 = (A2*lam2/2) * np.exp((1/2) * lam2 * (2*float(self.irf_mean) + lam2*(self.irf_width**2) - 2*x))
        term4 = erfc((float(self.irf_mean) + lam2*(self.irf_width**2) - x)/(self.irf_width*np.sqrt(2)))

        full = (term1*term2) + (term3*term4)
        return full

    def func_wrap(self, x, *params):
        p = np.array(params)

        match self.curve:
            case 'mono': return self.mono(x, p)
            case 'bi' | 'mh_bi' | 'nnls_bi' : return self.bi(x, p)
            case 'mono_conv' | 'mh_mono_conv': return self.mono_conv(x, p)
            case 'log_mono_conv': return self.log_mono_conv(x, p)
            case 'bi_conv' | 'nnls_bi_conv': return self.bi_conv(x, p)
            case _:
                raise Exception('Curve choice invalid in config.json, choose from mono, bi, mono_conv, and bi_conv.')



    '''Helper methods for Metropolis-Hastings'''
    def log_like(self, params, x, y, func):
        ym = func(x, params)
        return -0.5 * np.sum((y - ym)**2)

    def prop_lam(self, curr):
        return curr + np.random.normal(0, 5)

    def prop_A(self, curr):
        return curr + np.random.normal(0, 5000)

    def gibbs_mh(self, x, y, func, guess_params, iter=10000):
        curr_params = guess_params
        samples = []

        for _ in range(iter):
            for i in range(len(guess_params)):  
                if (i%2 == 0): prop = self.prop_A(curr_params[i])
                else: prop = self.prop_lam(curr_params[i])

                prop_params = curr_params
                prop_params[i] = prop

                prop_like = self.log_like(prop_params, x, y, func)
                curr_like = self.log_like(curr_params, x, y, func)

                if np.log(np.random.rand()) < (prop_like - curr_like):
                    curr_params[i] = prop

            samples.append(np.copy(curr_params))
        return np.array(samples)



    '''Deconvolution helper methods'''
    def gaussian(self, x, mu, sigma):
        kernel = np.exp(-(x-mu)**2/(2*sigma**2))
        kernel /= np.sum(kernel)
        return kernel

    def butter_lpf(self, data, cutoff=.1, order=2):
        fs = 1/(max(self.times)/len(self.times)) # sampling frequency
        nyq = 0.5*fs   
        cutoff_norm = cutoff/nyq

        b, a = butter(order, cutoff_norm, btype='low', analog=False)
        y = filtfilt(b, a, data)
        return y
    
    def deconvolve_fourier(self, alpha=1):
        data_filt = self.butter_lpf(self.data)
        data_filt /= np.sum(data_filt)
        F_data = fft(data_filt)
        
        irf = self.gaussian(self.times, self.irf_mean, self.irf_width)
        F_irf = fft(irf)

        F_dc = F_data * np.conj(F_irf) / (np.abs(F_irf)**2 + alpha**2)
        deconvolved = np.abs(ifft(F_dc))
        deconvolved /= np.sum(deconvolved)
        print(np.sum(deconvolved))

        plt.plot(self.times, self.data/np.sum(self.data), label='Original')
        plt.plot(self.times, data_filt, label='Filtered')
        plt.plot(self.times, deconvolved, label='Deconvolved')
        plt.plot(self.times, irf, label='irf')
        plt.legend()
        plt.show()

        return deconvolved
    


    '''Fitting main function'''
    def fit_decay(self):
        single = False
        
        match self.curve:
            case 'mono':
                loc = np.argmax(self.data)
                guess = [np.max(self.data), 0.1]
                # xdat = self.times[loc:]
                # ydat = self.data[loc:]
                xdat = self.times
                ydat = self.deconvolve_fourier(self.data/np.sum(self.data))
                
                single = True  
            case 'bi':
                loc = np.argmax(self.data)
                guess = [np.max(self.data), 0.1, np.max(self.data) / 2, 0.05]
                # xdat = self.times[loc:]
                # ydat = self.data[loc:]
                xdat = self.times
                ydat = self.deconvolve_fourier(self.data/np.sum(self.data))
            case 'mono_conv':
                guess = [np.max(self.data), 0.1]
                xdat = self.times
                ydat = self.data
                single = True
            case 'log_mono_conv':
                guess = [np.max(self.data), 2.0]
                xdat = self.times
                ydat = np.log(self.data + 1e-10)
                single = True
            case 'bi_conv':
                guess = [np.max(self.data), 4, np.max(self.data) / 2, 18]
                xdat = self.times
                ydat = self.data
            case 'mh_mono_conv':
                guess = np.array([1.0, 0.5])
                # samples = self.gibbs_mh(self.times, self.data, self.mono_conv, guess)
                # x = range(len(samples))
                # fig, ax = plt.subplots()
                # print(samples)
                # ax.plot(x, [x[0:] for x in samples])
                # plt.show()
                # # return (np.mean(samples[int(3*len(samples)/5):,0]), np.mean(samples[int(3*len(samples)/5):,1]), 0 ,0)
            case 'mh_bi':
                guess = [np.max(self.data), 0.1, np.max(self.data) / 2, 0.05]
                # loc = np.argmax(self.data)
                # samples = self.gibbs_mh(self.times[loc:], self.data[loc:], self.bi, guess)
                # return samples[-1]
            case 'nnls_bi':
                loc = np.argmax(self.data)
                guess = [np.max(self.data), 0.25, np.max(self.data) / 2, 0.05]
                xdat = self.times[loc:]
                ydat = self.data[loc:]

                params, _ = opt.curve_fit(self.func_wrap, xdat, ydat, p0=guess)
                
                A_d = np.vstack([np.exp(-params[1]*xdat), np.exp(-params[3]*xdat)]).T
                weights, _ = opt.nnls(A_d, ydat)

                return (weights[0], params[1], weights[1], params[3])
            case 'nnls_bi_conv':
                loc = np.argmax(self.data)
                guess = [np.max(self.data), 0.25, np.max(self.data) / 2, 0.05]
                xdat = self.times
                ydat = self.data

                params, _ = opt.curve_fit(self.func_wrap, xdat, ydat, p0=guess)
                
                A1_matrix = np.exp((1/2) * params[1] * (2 * float(self.irf_mean) + params[1] * (self.irf_width ** 2) - 2 * xdat)) * erfc((float(self.irf_mean) + params[1] * (self.irf_width ** 2) - xdat) / (self.irf_width * np.sqrt(2)))
                A2_matrix = np.exp((1/2) * params[3] * (2 * float(self.irf_mean) + params[3] * (self.irf_width ** 2) - 2 * xdat)) * erfc((float(self.irf_mean) + params[3] * (self.irf_width ** 2) - xdat) / (self.irf_width * np.sqrt(2)))
                A_d = np.vstack([A1_matrix, A2_matrix]).T

                weights, _ = opt.nnls(A_d, ydat)

                return (weights[0], params[1], weights[1], params[3])

            case _:
                raise Exception('Curve choice invalid in config.json, choose from mono, bi, mono_conv, and bi_conv.')

        params, _ = opt.curve_fit(self.func_wrap, xdat, ydat, p0=guess)

        if single:
            return (params[0], params[1], 0, 0)
        
        return params

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
            self.config['times'] = ((np.arange(numsteps) * step) + config['offset']) * 1e-3 # need times in ns
            self.config['numsteps'] = numsteps
            self.config['step'] = step * 1e-3 # need step in ns
        else:
            self.config['times'] = ((np.arange(config['numsteps']) * config['step']) + config['offset']) * 1e-3
            self.config['numsteps'] = config['numsteps']
            self.config['step'] = config['step'] * 1e-3

        self.A1 = None
        self.tau1 = None
        self.A2 = None
        self.tau2 = None
        self.intensity = None
        self.full_trace = None
        self.track = 0    

    @staticmethod
    def helper(config, data, i, j):
        length, x, y = np.shape(data)
        
        data_knl = np.zeros(length)
        for a in range(x):
            for b in range(y):
                data_knl += data[:, a, b]

        dt = Trace(config, data_knl, i, j)
        dt.fit_trace()
        return dt.params, dt.success, dt.sum, dt.i, dt.j



    '''Parallelizing helper function'''
    def fit_exps(self, filename=None, image=None):
        tic = time.time()
        print('Reading image')
        if filename:
            with tf.TiffFile(filename + '.tif') as tif:
                image = tif.asarray(key=range(self.config['numsteps']))  # Only read the first 5000 frames
            length, x, y = np.shape(image)
        elif image is not None:
            image = image[:self.config['numsteps'],:,:]
            length, x, y = np.shape(image)
        else:
            raise Exception('No filename or image provided to fit_exps, make sure to provide one or the other.')
        toc = time.time()
        print(f'Image read in {toc-tic} seconds')

        self.A1 = np.zeros((x, y), dtype=float)
        self.A2 = np.zeros((x, y), dtype=float)
        self.tau1 = np.zeros((x, y), dtype=float)
        self.tau2 = np.zeros((x, y), dtype=float)
        self.intensity = np.zeros((x, y), dtype=float)
        self.full_trace = np.zeros((self.config['numsteps']), dtype=float)

        ksize = self.config['kernel_size']
        with ProcessPoolExecutor() as executor:
            futures = [executor.submit(self.helper, self.config, image[:, (i-ksize):(i+ksize+1), (j-ksize):(j+ksize+1)], i, j) for i in range(ksize,x-ksize) for j in range(ksize, y-ksize)]
            # futures = [executor.submit(self.helper, self.config, image[:, (i-ksize):(i+ksize+1), (j-ksize):(j+ksize+1)], i, j) for i in range(179, 181) for j in range(171, 173)]
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
                    print(f'Pixel ({i}, {j}): {1/(outputs[1]+1e-10)} ns\n')

        print(len(self.full_trace))
        full_reshaped = self.full_trace.reshape(len(self.full_trace),1,1)

        outputs, success, sum, i, j = self.helper(self.config, full_reshaped, 0, 0)

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

        self.config['step'] = self.config['step'] * 1e3 # reverse change made in initializaiton