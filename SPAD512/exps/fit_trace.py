import numpy as np
from scipy import optimize as opt
from scipy.signal import convolve, deconvolve, butter, filtfilt
from scipy.fft import fft, ifft, fftfreq
from scipy.special import erfc
import matplotlib.pyplot as plt
import warnings

class Trace:
    def __init__(self,data,i,j,**kwargs):
        # see sims/gen_single.py for explanation of this class setup logic
        defaults = {
            'step': 0,
            'fit': "",
            'irf_width': 0,
            'irf_mean': 0,
            'thresh': 0,
            'width': 0,
            'times': 0,
            'offset': 0,
            'bits': 5,
            'integ': 10000,
            'freq': 10,
            'kernel_size': 3,
            'track': 0
        }
        filtered = {k: v for k, v in kwargs.items() if k in defaults}
        defaults.update(filtered)

        for key, val in defaults.items():
            setattr(self,key,val)

        self.data = data
        self.i = i
        self.j = j # assign pixel values as parameters so that helper method can return them
        
        self.success = False 
        self.sum = np.sum(self.data)
    
    '''Fitting functions'''
    def mono(self, x, *p): # mono exponential
        A, lam = p
        return A * np.exp(-lam * x)

    def mono_conv(self, x, *p): # mono gaussian-convolved-exponential
        A, lam = p
        term1 = (A*lam/2) * np.exp((1/2) * lam * (2*float(self.irf_mean) + lam*(self.irf_width**2) - 2*x))
        term2 = erfc((float(self.irf_mean) + lam*(self.irf_width**2) - x)/(self.irf_width*np.sqrt(2)))
        return term1*term2

    def log_mono_conv(self, x, *p): # natural log scale mono_conv
        A, lam = p
        if (A <= 0):
            A = 1e-10
        if (lam <= 0):
            lam = 1e-10
        
        term1 = np.log(A*lam/2) + ((1/2) * lam * (2*float(self.irf_mean) + lam*(self.irf_width**2) - 2*x))
        term2 = np.log(erfc((float(self.irf_mean) + lam*(self.irf_width**2) - x)/(self.irf_width*np.sqrt(2))))
        return term1 + term2
        
    def bi(self, x, *p): # bi-exponential
        A, lam1, B, lam2 = p
        return A *(B * np.exp(-x * lam1) + (1-B) * np.exp(-x * lam2))

    def bi_conv(self, x, *p): # gaussian convolved bi-exponential
        A1, lam1, A2, lam2 = p

        term1 = (A1*lam1/2) * np.exp((1/2) * lam1 * (2*float(self.irf_mean) + lam1*(self.irf_width**2) - 2*x))
        term2 = erfc((float(self.irf_mean) + lam1*(self.irf_width**2) - x)/(self.irf_width*np.sqrt(2)))
        term3 = (A2*lam2/2) * np.exp((1/2) * lam2 * (2*float(self.irf_mean) + lam2*(self.irf_width**2) - 2*x))
        term4 = erfc((float(self.irf_mean) + lam2*(self.irf_width**2) - x)/(self.irf_width*np.sqrt(2)))

        full = (term1*term2) + (term3*term4)
        return full



    '''Helper methods for Metropolis-Hastings'''
    def log_like(self, params, x, y, func): # get log likelihood of model match
        ym = func(x, params)
        return -0.5 * np.sum((y - ym)**2)

    def prop_lam(self, curr): # propose a lifetime
        return curr + np.random.normal(0, 5)

    def prop_A(self, curr): # propose an amplitude
        return curr + np.random.normal(0, 5000)

    def gibbs_mh(self, x, y, func, guess_params, iter=10000): #gibbs sampling scheme to coordinate metropolis hastings "fitting"
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
    def gaussian(self, x, mu, sigma): # define an IRF 
        kernel = np.exp(-(x-mu)**2/(2*sigma**2))
        kernel /= np.sum(kernel)
        return kernel

    def butter_lpf(self, data, cutoff=.1, order=2): # pass filter to make the function buttery smooth for fourier deconvolution
        fs = 1/(max(self.times)/len(self.times)) # sampling frequency
        nyq = 0.5*fs   
        cutoff_norm = cutoff/nyq

        b, a = butter(order, cutoff_norm, btype='low', analog=False)
        y = filtfilt(b, a, data)
        return y
    
    def deconvolve_fourier(self, alpha=1): # fourier deconvolution, doesn't really work
        data_filt = self.butter_lpf(self.data)
        data_filt /= np.sum(data_filt)
        F_data = fft(data_filt)
        
        irf = self.gaussian(self.times, self.irf_mean, self.irf_width)
        F_irf = fft(irf)

        F_dc = F_data * np.conj(F_irf) / (np.abs(F_irf)**2 + alpha**2)
        deconvolved = np.abs(ifft(F_dc))
        deconvolved /= np.sum(deconvolved)

        return deconvolved
    
    # # another failed attempt at writing deconvolution code AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
    # def deconvolve_RL(self, iter=50):
    #     data = self.data.astype(np.float64)
    #     psf = self.gaussian(self.times, self.irf_mean, self.irf_width)
    #     psf /= psf.sum()  
    #     psf_mirror = psf[::-1]
    #     est = np.full_like(data, np.mean(data) + 1e-10)
        
    #     for _ in range(iter):
    #         convolved_est = convolve(est, psf, mode='same') + 1e-10
    #         blur = data / convolved_est
    #         est *= convolve(blur, psf_mirror, mode='same')
    #         est = np.maximum(est, 0)
        
    #     return est
    


    '''correction formulas'''
    def correct(self, antol=False, save=False): # interpolate counts into the raw detection probability
        raw_data = self.data

        if antol:
            max_counts = ((1 + self.kernel_size*2)**2) * (2**self.bits - 1)

            probs = self.data/max_counts
            self.data = -max_counts * np.log(1 - probs + 1e-9) # avoid ln(0) with an epsilon
        else: 
            bin_time = self.integ/(2**self.bits - 1)
            bin_gates = int(self.freq * bin_time)
            
            max_counts = ((1 + self.kernel_size*2)**2) * (2**self.bits - 1)
                
            probs = self.data/max_counts
            probs = 1 - (1 - probs)**(1/(bin_gates))
            self.data = max_counts*probs # rescale            

    '''Fitting main function'''
    def fit_decay(self, full=False): # organize fitting logic based on what fit was chosen in config
        warnings.filterwarnings('ignore', category=RuntimeWarning) # ignore warnings when doing NN-NL-LS
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
                xdat = self.times[loc:]
                ydat = self.data[loc:]

                A_guess = np.max(ydat)
                lam1_guess = 1.0 / (xdat[0] + 1e-6)  
                lam2_guess = lam1_guess / 2
                B_guess = 0.5  # maybe there's a better guess here

                guess = [A_guess, lam1_guess, B_guess, lam2_guess]
                bounds = (
                    [0, 1e-8, 0, 1e-8],  # enforce strict positivity for all parameters
                    [np.inf, np.inf, 1, np.inf]  # B <= 1
                )

                def residuals(p, x, y, cutoff = 1e-2):
                    A, lam1, B, lam2 = p
                    model = self.bi(x, A, lam1, B, lam2)
                    res = model - y
                    
                    delta_lam = np.abs(lam1 - lam2) # penalize close lambdas
                    if delta_lam < cutoff:
                        penalty = 1e6 * (cutoff - delta_lam)
                        res += penalty

                    return res

                try:
                    # lma doesn't work with bounds, so use trf, but maybe dogbox would work better?
                    results = opt.least_squares(residuals, guess, args=(xdat, ydat), bounds=bounds, method='trf')
                    params = results.x

                    return params

                except Exception as e:
                    raise RuntimeError("Fitting failed: " + str(e))

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

                try:
                    params, _ = opt.curve_fit(self.bi, xdat, ydat, p0=guess)
                    A_d = np.vstack([np.exp(-params[1]*xdat), np.exp(-params[3]*xdat)]).T

                    cond_num = np.linalg.cond(A_d) # check matrix condition and add regularization if needed
                    if cond_num > 1e12:
                        print("Warning: Matrix is ill-conditioned, adding regularization.")
                        A_d += (1e-6) * np.identity(A_d.shape[1])

                    weights, _ = opt.nnls(A_d, ydat)
                    params[0] = weights[0] + weights[1]
                    params[2] = weights[0] / params[0]    
                    return params         
                
                except np.linalg.LinAlgError as e:
                    raise RuntimeError("Matrix is singular, unable to proceed with NNLS.")

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

                return (A1, 1/tau1, A2, 1/tau2)

            case 'stref': # untested
                loc = min(np.argmax(self.data), len(self.data) - 4)
                guess = [np.max(self.data), 0.25, np.max(self.data) / 2, 0.05]
                xdat = self.times
                ydat = self.data

            case _:
                raise Exception('Curve choice invalid in config.json, choose from mono, mono_conv, mono_conv_log, mono_conv_mh, bi, bi_nnls, bi_conv, bi_conv_nnls, mono_rld, mono_rld_50ovp, bi_rld')

    def fit_trace(self, full=False): # dont fit pixels that dont meet the specified count threshold
        if self.sum > self.thresh:
            try:
                if not full: self.correct(antol=True)
                self.params = list(self.fit_decay(full=full))
                self.success = True

                # newdat = self.bi(self.times, *self.params)
                # newdat = 255*newdat/max(newdat)
                # plt.plot(self.times, newdat, 'k-o', label=f'Uncorrected - Tau1: {1/self.params[1]:.2f}, Tau2: {1/self.params[3]:.2f}')

                # # if not full: self.correct(antol=True)
                # # self.params = list(self.fit_decay(full=full))
                # # self.success = True

                # # newdat = self.bi(self.times, *self.params)
                # # plt.plot(self.times, 255*newdat/max(newdat), 'b-o', label=f'Corrected - Tau1: {1/self.params[1]:.2f}, Tau2: {1/self.params[3]:.2f}')
                # # plt.legend()
                # # plt.grid(True)
                # # plt.show()

                if self.params[1] > self.params[3]:
                    temp = self.params[3]
                    self.params[3] = self.params[1]
                    self.params[1] = temp
                    self.params[2] = 1 - self.params[2] # swap B weight because taus were swapped

                if not full: np.savez(r'C:\Users\ishaa\Documents\FLIM\figure_remaking\figure2_500us_corrected-trace',
                                    times=self.times,
                                    data=self.data,
                                    params=self.params)

            except (RuntimeError, ValueError):
                self.params = [0, 0, 0, 0]
        else: 
            self.params = [0, 0, 0, 0]

def comp_prob(t, lts, wgt, zeta, width):
    p = np.zeros(len(t))
    for i, lt in enumerate(lts):
        lam = 1 / lt
        p += wgt[i] * zeta * (np.exp(-lam * t) - np.exp(-lam * (t + width)))
    return p

if __name__ == '__main__':
    bits = 8
    freq = 10 
    width_ps = 10000  
    step_ps = 500     
    offset_ps = 5000  
    zeta = 0.05
    dark_cps = 25

    lts_true = [20, 5] 
    wgt_true = [0.5, 0.5]

    width = width_ps * 1e-3  
    step = step_ps * 1e-3   
    offset = offset_ps * 1e-3  
    total_img = 2 ** bits - 1
    numsteps = int(1e3 / (freq * step))
    times = np.arange(numsteps) * step + offset

    prob_true = comp_prob(times, lts_true, wgt_true, zeta, width)

    integs = [1000, 2500, 5000, 7500, 10000, 12500, 15000, 17500, 20000]

    tau1_errors = []
    tau2_errors = []
    integ_times_s = []  

    lam1_true = 1 / lts_true[0]
    lam2_true = 1 / lts_true[1]

    for integ in integs:
        integ_s = integ / 1e3
        integ_times_s.append(integ_s)

        numgates = freq * integ_s * 1e3  
        bin_gates = numgates / total_img
    
        P_bin = 1 - (1 - prob_true) ** bin_gates
        counts = np.random.binomial(total_img, P_bin)

        trace = Trace(data=counts, i=0, j=0, times=times, fit='bi', step=step)

        try:
            trace.fit_trace()
            if trace.success:
                params_estimated = trace.params
                A_hat, lam1_hat, B_hat, lam2_hat = params_estimated
                tau1_hat = 1 / lam1_hat
                tau2_hat = 1 / lam2_hat

                tau1_error = tau1_hat - lts_true[0]
                tau2_error = tau2_hat - lts_true[1]
            else:
                tau1_error = np.nan
                tau2_error = np.nan
        except Exception as e:
            print(f"Fit failed at integration time {integ_s:.2f}s: {e}")
            tau1_error = np.nan
            tau2_error = np.nan

        tau1_errors.append(tau1_error)
        tau2_errors.append(tau2_error)

    integ_times_s = np.array(integ_times_s)
    tau1_errors = np.array(tau1_errors)
    tau2_errors = np.array(tau2_errors)

    plt.figure(figsize=(10, 6))
    plt.plot(integ_times_s, tau1_errors, 'o-', label='Tau1 Error')
    plt.plot(integ_times_s, tau2_errors, 's-', label='Tau2 Error')
    plt.xlabel('Integration Time (s)')
    plt.ylabel('Lifetime Estimation Error (ns)')
    plt.title('Effect of Integration Time on Lifetime Estimates')
    plt.legend()
    plt.grid(True)
    plt.show()