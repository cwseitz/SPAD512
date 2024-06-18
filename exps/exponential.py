import numpy as np
import time
import scipy.optimize as opt
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from multiprocessing import Pool
from skimage.io import imsave, imread
from scipy.io import savemat
from scipy.signal import convolve, deconvolve

'''
Script to fit deconvolved exponentials accross a widefield FLIM image and display the full results. 
To-do: 
- Add deconvolution code
- Automate threshold determination
- Choose better start point for exponential fitting
'''

# define parameters
filename = '240604/240604_10ms_adjusted.tif'
thresh = 10000
gate_step = 0.09 # gate step size in ns
gate_width = 5 # gate width in ns
gate_num = 1000 # number of gates for a given step
gate_offset = 0.018 # gate offset in ns
irf_width = 0

# exponential decay helper function
def decay(x, amp, tau):
        return amp * np.exp(-x / tau)

# scipy.curve_fit wrapper
def fit_decay(times, data):
        initial_guess = [np.max(data), 2.0] # use max for the amplitude, choice is ultimately insignificant with LMA algorithm
        params, cov = opt.curve_fit(decay, times, data, p0=initial_guess)
        return params

# deconvolution function
def irf_deconvolve(times, trace, irf_width):
    irf = np.exp(-times**2 / irf_width)
    tracedc = deconvolve(trace, irf)
    return tracedc

# read image and set initial arrays
image = imread(filename)
length, x, y = np.shape(image)
times = (np.arange(length) * gate_step) + gate_offset
A = np.zeros((x, y))
intensity = np.zeros((x, y))
tau = np.zeros((x, y))

# fit exponentials and save info
for i in range(x):
    for j in range(y):
        trace = image[:, i, j]
        if (sum(trace) > thresh):
            intensity[i][j] += sum(trace)
            loc = np.argmax(trace) # only fit from exponential peak onwards
            params = fit_decay(times[loc:], trace[loc:])
            A[i][j] += params[0]
            tau[i][j] += params[1]
            print(f'Lifetime at pixel {i}, {j}: {params[1]}')

            # plt.figure(figsize=(6, 4))
            # plt.scatter(times, trace, s=5, label='Data')
            # plt.plot(times, decay(times, params[0], params[1]), label='Fit: tau = {:.2f}'.format(params[1]), color='black')
            # plt.xlabel('Time, ns')
            # plt.ylabel('Counts')
            # plt.legend()
            # plt.title('Simulated Decay for 1 ms integration, 90 ps step')
            # plt.show()

# plot code taken from replot.py
for i in range(len(tau)):
  for j in range(len(tau[0])):
    if tau[i][j] > 1000:
      tau[i][j] = 0
    if tau[i][j] < -1000:
       tau[i][j] = 0

fig,ax=plt.subplots(1,3,figsize=(9,3),sharex=True,sharey=True)
im1 = ax[0].imshow(A,cmap='plasma')
im2 = ax[1].imshow(intensity,cmap='gray')
im3 = ax[2].imshow(tau,cmap='hsv')
ax[0].set_title('A')
ax[1].set_title('Intensity')
ax[2].set_title(r'$\tau$')

for axi in ax.ravel():
    axi.set_xticks([])
    axi.set_yticks([])

plt.colorbar(im1,ax=ax[0],label='cts')
plt.colorbar(im2,ax=ax[1],label='cts')
plt.colorbar(im3,ax=ax[2],label='ns')
plt.tight_layout()
plt.show()
