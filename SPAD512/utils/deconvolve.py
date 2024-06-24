import numpy as np
from scipy.signal import convolve, deconvolve
from scipy.special import eval_genlaguerre
import matplotlib.pyplot as plt


'''
Test script for wiener deconvolution of gaussians from exponential
'''

length = 1000
time = np.linspace(0, 10, length)
tau = 2.0
decay = np.exp(-time / tau)

irf_width = 5
irf = np.exp(-time**2 / irf_width)
irf /= np.sum(irf)  # normalize

detected = convolve(decay, irf, mode='full') / irf.sum()
detected = detected[:length]

'''wiener deconvolution'''
# snr = 1e-2

# traceft = np.fft.fft(detected)
# irfft = np.fft.fft(irf, n=len(detected))


# kp = np.abs(irfft)**2
# filter = np.conj(irfft) / (kp + 1 / snr)

# dcfft = traceft * filter
# deconvolved = np.fft.ifft(dcfft)

'''richardson lucy deconvolution'''
# iter = 50000
# deconvolved = np.full(decay.shape, 0.9)
# for i in range(iter):
#     rel_blur = decay / convolve(deconvolved, irf, mode='same')
#     deconvolved *= convolve(rel_blur, irf[::-1], mode = 'same')
#     print(i)

# '''laguerre basis'''
# def basis(n, alpha, t):
#     return np.exp(-alpha * t) * eval_genlaguerre(n, alpha, t)

# def transform(signal, n_basis, alpha):
#     t = np.arange(len(signal))
#     coeffs = np.zeros(n_basis)
#     for n in range(n_basis):
#         L_n = basis(n, alpha, t)
#         coeffs[n] = np.dot(signal, L_n)
#     return coeffs

# def inv_transform(coeffs, n_basis, alpha, length):
#     t = np.arange(length)
#     signal = np.zeros(length)
#     for n in range(n_basis):
#         L_n = basis(n, alpha, t)
#         signal += coeffs[n] * L_n
#     return signal

# def deconvlag(signal, irf, n_basis=10, alpha=0.5):
#     signal_coeffs = transform(signal, n_basis, alpha)
#     irf_coeffs = transform(irf, n_basis, alpha)

#     deconvolved_coeffs = np.zeros(n_basis)
#     for n in range(n_basis):
#         if irf_coeffs[n] != 0:
#             deconvolved_coeffs[n] = signal_coeffs[n] / irf_coeffs[n]

#     deconvolved_signal = inv_transform(deconvolved_coeffs, n_basis, alpha, len(signal))
#     return deconvolved_signal

# deconvolved = deconvlag(detected, irf)

'''richardson lucy total variation'''
import numpy as np
from scipy.signal import convolve
from scipy.ndimage import gaussian_filter1d

def tv_denoise(signal, weight, iterations=10):
    """Perform total variation denoising."""
    for _ in range(iterations):
        grad = np.gradient(signal)
        grad_norm = np.sqrt(grad**2 + 1e-8)
        signal -= weight * grad / grad_norm
    return signal

def richardson_lucy_tv_deconvolution(signal, psf, iterations=50, tv_weight=0.01, tv_iterations=10):
    psf_mirror = psf[::-1]
    estimate = np.full(signal.shape, 0.5)

    for _ in range(iterations):
        relative_blur = signal / (convolve(estimate, psf, mode='same') + 1e-8)
        estimate *= convolve(relative_blur, psf_mirror, mode='same')
        estimate = tv_denoise(estimate, tv_weight, tv_iterations)
    
    return estimate

deconvolved = richardson_lucy_tv_deconvolution(detected, irf, iterations=100, tv_weight=0.001, tv_iterations=50)


# plot results
plt.plot(time, decay, label='Original Trace')
plt.plot(time, detected, label='Convolved Signal')
plt.plot(time, deconvolved, label='Deconvolved Signal')
plt.xlabel('Time (ns)')
plt.ylabel('Intensity')
plt.legend()
plt.show()