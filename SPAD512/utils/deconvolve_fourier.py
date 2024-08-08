import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import fftconvolve, convolve, deconvolve, butter, filtfilt
from scipy.fft import fft, ifft, fftfreq


'''Deconvolution helper methods'''
def gaussian(x, mu, sigma):
    kernel = np.exp(-(x-mu)**2/(2*sigma**2))
    kernel /= np.sum(kernel)
    return kernel

def butter_lpf(self, data, cutoff, fs, order):
    T = 1 # sample period
    fs = 1/(max(self.times)/len(self.times)) # sampling frequency
    cutoff = 2

    nyq = 0.5*fs
    order = 2
    n = int(T*fs)   
    cutoff_norm = cutoff/nyq

    b, a = butter(order, cutoff_norm, btype='low', analog=False)
    y = filtfilt(b, a, data)
    return y

# def deconvolve_fourier(self, alpha=1):
#     data_filt = self.butter_lpf(self.data)
#     F_data = fft(data_filt)
    
#     irf = self.gaussian(self.times, self.irf_mean, self.irf_width)
#     F_irf = fft(irf)

#     F_dc = F_data * np.conj(F_irf) / (np.abs(F_irf)**2 + alpha**2)
#     deconvolved = ifft(F_dc)
#     deconvolved /= np.sum(deconvolved)

#     return deconvolved

length = 900
time = np.linspace(0, 90, length)
time = time + 0.018
tau = 10
decay = 100* np.exp(-time / tau)
decay /= np.sum(decay)

irf_width = 1
irf_mean = 10
irf = gaussian(time, irf_mean, irf_width)
irf /= np.sum(irf)  # normalize

detected = convolve(decay, irf, mode='full') / irf.sum()
detected = detected[:length]

'''deconvolution'''
F_observed = fft(detected)
F_gaussian = fft(irf)

alpha = 1

F_deconvolved = F_observed * np.conj(F_gaussian) / (np.abs(F_gaussian)**2 + alpha**2)

deconvolved = ifft(F_deconvolved)
normed = deconvolved/(np.sum(deconvolved))



plt.plot(time, decay, label='Original Trace, tau = 10 ns')
plt.plot(time, detected, label='Convolved Signal')
# plt.plot(time, irf, label='IRF, N(10, 1)')
plt.plot(time, normed, label='Deconvolved Signal')
plt.xlabel('Time (ns)')
plt.ylabel('Intensity')
plt.legend()
plt.show()
