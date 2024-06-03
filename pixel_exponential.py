import numpy as np
import tifffile as tf
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

'''
Script for pixel by pixel exponential fitting.
To-do list:
- Verify that exponential fitting is actually being done correctly (maybe need to normalize and change function)
- Determine cutoff values for variance (when to discard the beta calculated in a pixel)
'''

filename = '240529_SPAD-QD-20MHz-5f-1000g-5ns-18ps-18ps-150uW.tif'
freq = 20 # Mhz
frames = 5 
perframe = 1000 # number of gates per frame
width = 5 # ns
step = 0.018 # ns
offset = 0.018 # ns
integ = 10 # ms

# read whole tiff image
data = tf.TiffFile(filename)
data = data.asarray()
images, xlen, ylen = data.shape

#create timeseries for histogram (nanoseconds)
times = np.zeros(int(len(data)/frames))
for i in range(len(times)):
    times[i] += (width/2) + i*(step) + offset

# helper function for exponential fitting with scipy.optimize.curve_fit
def fitFunc(t, a, b):
    return a*np.exp(-b*t)

# create an exponential fit for each pixel and assign beta value to a new array to be plotted
betas = np.zeros((xlen, ylen))
err = np.zeros((xlen, ylen))
for i in range(xlen):
    for j in range(ylen):
        counts = np.zeros((len(times)))
        for p in range(frames):
            counts += data[(perframe*p):(perframe + perframe*p), i, j]
        params, cov = curve_fit(fitFunc, times, counts)
        betas[i, j] = 1/params[1]
        if (abs(betas[i,j]) > 1 or betas[i,j] < 0):
            betas[i,j] = 0
        err[i,j] = cov[1, 1]    


plt.imshow(betas, cmap='hot', interpolation='nearest')
plt.colorbar(label='Decay Rate')
plt.title('Heat map of decay rates accross pixels')
plt.show()
