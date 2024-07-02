import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import fftconvolve, convolve, deconvolve
from scipy.fft import fft, ifft, fftfreq



'''data creation'''
def gaussian_kernel(x, mu, sigma):
    kernel = np.exp(-(x-mu)**2 / (2 * sigma**2)) 
    kernel /= np.sum(kernel)
    return kernel

length = 900
time = np.linspace(0, 90, length)
time = time + 0.018
tau = 10
decay = 100* np.exp(-time / tau)
decay /= np.sum(decay)

irf_width = 3
irf_mean = 10
irf = gaussian_kernel(time, irf_mean, irf_width)
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

print(decay)
print(detected)
print(deconvolved)

plt.plot(time, decay, label='Original Trace')
plt.plot(time, detected, label='Convolved Signal')
plt.plot(time, deconvolved, label='Deconvolved Signal')
plt.plot(time, irf, label='IRF')
plt.plot(time, normed, label='Normalized Deconvolved')
plt.xlabel('Time (ns)')
plt.ylabel('Intensity')
plt.legend()
plt.show()
