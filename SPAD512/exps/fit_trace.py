import numpy as np
from scipy import optimize as opt
from scipy.signal import convolve, deconvolve, butter, filtfilt
from scipy.fft import fft, ifft, fftfreq
from scipy.special import erfc
import matplotlib.pyplot as plt
import warnings

class Trace:
    def __init__(self,data,i,j,**kwargs):
        defaults = {
            'step': 0,
            'fit': "",
            'irf_width': 0,
            'irf_mean': 0,
            'thresh': 0,
            'width': 0,
            'times': 0,
            'offset': 0
        }
        filtered = {k: v for k, v in kwargs.items() if k in defaults}
        defaults.update(filtered)

        for key, val in defaults.items():
            setattr(self,key,val)

        self.data = data
        self.i = i
        self.j = j
        
        self.success = False
        self.sum = np.sum(self.data)
    


    '''Fitting functions'''
    def mono(self, x, *p):
        A, lam = p
        return A * np.exp(-lam * x)

    def mono_conv(self, x, *p):
        A, lam = p
        term1 = (A*lam/2) * np.exp((1/2) * lam * (2*float(self.irf_mean) + lam*(self.irf_width**2) - 2*x))
        term2 = erfc((float(self.irf_mean) + lam*(self.irf_width**2) - x)/(self.irf_width*np.sqrt(2)))
        return term1*term2

    def log_mono_conv(self, x, *p):
        A, lam = p
        if (A <= 0):
            A = 1e-10
        if (lam <= 0):
            lam = 1e-10
        
        term1 = np.log(A*lam/2) + ((1/2) * lam * (2*float(self.irf_mean) + lam*(self.irf_width**2) - 2*x))
        term2 = np.log(erfc((float(self.irf_mean) + lam*(self.irf_width**2) - x)/(self.irf_width*np.sqrt(2))))
        return term1 + term2
        
    def bi(self, x, *p):
        A, lam1, B, lam2 = p
        return A *(B * np.exp(-x * lam1) + (1-B) * np.exp(-x * lam2))

    def bi_conv(self, x, *p):
        A1, lam1, A2, lam2 = p

        term1 = (A1*lam1/2) * np.exp((1/2) * lam1 * (2*float(self.irf_mean) + lam1*(self.irf_width**2) - 2*x))
        term2 = erfc((float(self.irf_mean) + lam1*(self.irf_width**2) - x)/(self.irf_width*np.sqrt(2)))
        term3 = (A2*lam2/2) * np.exp((1/2) * lam2 * (2*float(self.irf_mean) + lam2*(self.irf_width**2) - 2*x))
        term4 = erfc((float(self.irf_mean) + lam2*(self.irf_width**2) - x)/(self.irf_width*np.sqrt(2)))

        full = (term1*term2) + (term3*term4)
        return full



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

        # plt.plot(self.times, self.data/np.sum(self.data), label='Original')
        # plt.plot(self.times, data_filt, label='Filtered')
        # plt.plot(self.times, deconvolved, label='Deconvolved')
        # plt.plot(self.times, irf, label='irf')
        # plt.legend()
        # plt.show()

        return deconvolved
    


    '''Fitting main function'''
    def fit_decay(self):
        warnings.filterwarnings('ignore', category=RuntimeWarning)
        warnings.filterwarnings('ignore', category=opt.OptimizeWarning)

        match self.fit:
            case 'mono':
                loc = min(np.argmax(self.data), len(self.data) - 2)
                guess = [np.max(self.data), 0.1]
                xdat = self.times[loc:]
                ydat = self.data[loc:]
                # xdat = self.times
                # ydat = self.deconvolve_fourier(self.data/np.sum(self.data))
                params, _ = opt.curve_fit(self.mono, xdat, ydat, p0=guess)
                return (params[0], params[1], 0, 0) 

            case 'bi':
                loc = min(np.argmax(self.data), len(self.data) - 4)
                guess = [np.max(self.data), 0.1, np.max(self.data) / 2, 0.05]
                xdat = self.times[loc:]
                ydat = self.data[loc:]
                # xdat = self.times
                # ydat = self.deconvolve_fourier(self.data/np.sum(self.data))
                params, _ = opt.curve_fit(self.bi, xdat, ydat, p0=guess)
                return params

            case 'mono_conv':
                guess = [np.max(self.data), 0.1]
                xdat = self.times
                ydat = self.data
                params, _ = opt.curve_fit(self.mono_conv, xdat, ydat, p0=guess)
                return (params[0], params[1], 0, 0) 

            case 'mono_conv_log':
                guess = [np.max(self.data), 2.0]
                xdat = self.times
                ydat = np.log(self.data + 1e-10)
                params, _ = opt.curve_fit(self.log_mono_conv, xdat, ydat, p0=guess)
                return (params[0], params[1], 0, 0) 

            case 'bi_conv':
                guess = [np.max(self.data), 4, np.max(self.data) / 2, 18]
                xdat = self.times
                ydat = self.data
                params, _ = opt.curve_fit(self.bi_conv, xdat, ydat, p0=guess)
                return params

            case 'mono_conv_mh':
                guess = np.array([1.0, 0.5])
                samples = self.gibbs_mh(self.times, self.data, self.mono_conv, guess)
                return samples[-1]

            case 'bi_mh':
                guess = [np.max(self.data), 0.1, np.max(self.data) / 2, 0.05]
                loc = np.argmax(self.data)
                samples = self.gibbs_mh(self.times[loc:], self.data[loc:], self.bi, guess)
                return samples[-1]

            case 'bi_nnls':
                loc = min(np.argmax(self.data), len(self.data) - 4)
                guess = [np.max(self.data), 0.25, np.max(self.data) / 2, 0.05]
                xdat = self.times[loc:]
                ydat = self.data[loc:]

                params, _ = opt.curve_fit(self.bi, xdat, ydat, p0=guess)
                
                A_d = np.vstack([np.exp(-params[1]*xdat), np.exp(-params[3]*xdat)]).T
                weights, _ = opt.nnls(A_d, ydat)

                return (weights[0], params[1], weights[1], params[3])

            case 'bi_conv_nnls':
                loc = min(np.argmax(self.data), len(self.data) - 4)
                guess = [np.max(self.data), 0.25, np.max(self.data) / 2, 0.05]
                xdat = self.times
                ydat = self.data

                params, _ = opt.curve_fit(self.bi_conv, xdat, ydat, p0=guess)
                
                A1_matrix = np.exp((1/2) * params[1] * (2 * float(self.irf_mean) + params[1] * (self.irf_width ** 2) - 2 * xdat)) * erfc((float(self.irf_mean) + params[1] * (self.irf_width ** 2) - xdat) / (self.irf_width * np.sqrt(2)))
                A2_matrix = np.exp((1/2) * params[3] * (2 * float(self.irf_mean) + params[3] * (self.irf_width ** 2) - 2 * xdat)) * erfc((float(self.irf_mean) + params[3] * (self.irf_width ** 2) - xdat) / (self.irf_width * np.sqrt(2)))
                A_d = np.vstack([A1_matrix, A2_matrix]).T

                weights, _ = opt.nnls(A_d, ydat)

                return (weights[0], params[1], weights[1], params[3])

            case 'mono_rld':
                D0, D1 = self.data
                A = (D0**2) * (np.log(D0/D1)) / (self.step*(D0-D1))
                tau = self.step / (np.log(D0/D1))
                return (A, 1/tau, 0, 0)

            case 'mono_rld_50ovp':
                self.step /= 1000
                D0, D1 = self.data
                A = 2 * (D0**3) * (np.log(D1/D0)) / (self.step*((D1**2)-(D0**2)))
                tau = -self.step / (np.log((D1**2)/(D0**2)))
                return (A, 1/tau, 0, 0)

            case 'bi_rld': 
                D0, D1, D2, D3 = self.data
                dt = self.step
                g = self.width

                R = D1*D1 - D2*D0
                P = D3*D0 - D2*D1
                Q = D2*D2 - D3*D1
                disc = P**2 - 4*R*Q
                y = (-P + np.sqrt(disc))/(2*R)
                x = (-P - np.sqrt(disc))/(2*R)
                S = self.step * ((x**2)*D0 - (2*x*D1) + D2)
                T = (1-((x*D1 - D2)/(x*D0 - D1))) ** (g/dt)

                tau1 = -dt/np.log(y)
                tau2 = -dt/np.log(x)
                A1 = (-(x*D0 - D1)**2) * np.log(y) / (S * T) 
                A2 = (-R * np.log(x)) / (S * ((x**(g/dt)) - 1))

                return (A1, A2, 1/tau1, 1/tau2)

            case 'stref':
                loc = min(np.argmax(self.data), len(self.data) - 4)
                guess = [np.max(self.data), 0.25, np.max(self.data) / 2, 0.05]
                xdat = self.times
                ydat = self.data

            case _:
                raise Exception('Curve choice invalid in config.json, choose from mono, mono_conv, mono_conv_log, mono_conv_mh, bi, bi_nnls, bi_conv, bi_conv_nnls, mono_rld, mono_rld_50ovp, bi_rld')

    def fit_trace(self):
        if self.sum > self.thresh:
            try:
                self.params = self.fit_decay()
                self.success = True
            except RuntimeError:
                self.params = [0, 0, 0, 0]
        else: self.params = [0, 0, 0, 0]