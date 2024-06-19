import numpy as np
from scipy.signal import convolve, deconvolve
import matplotlib.pyplot as plt

'''
Test script for deconvolution of gaussians from exponential
'''

time = np.linspace(0, 10, 1000)
tau = 2.0
decay = np.exp(-time / tau)

irf_width = 0.1
irf = np.exp(-time**2 / irf_width)
irf /= np.sum(irf)  # normalize

detected = convolve(decay, irf, mode='full') / irf.sum()
deconvolved, remainder = deconvolve(detected, irf)
deconvolved *= irf.sum()

# print(np.shape(decay))
# print(np.shape(irf))
# print(np.shape(detected))
# print(np.shape(deconvolved))

# plot results
plt.figure(figsize=(12, 6))
plt.plot(time, detected[500:1500], label='Measured Signal')
plt.plot(time, decay, label='True Signal')
plt.plot(time, deconvolved, label='Deconvolved Signal', linestyle='--')
plt.xlabel('Time')
plt.ylabel('Intensity')
plt.title('Deconvolution of Gaussian IRF from Exponential Decay Signal')
plt.legend()
plt.show()