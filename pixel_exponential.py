import numpy as np
import tifffile as tf
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

'''
Script for pixel by pixel exponential fitting.
To-do list:
- Fix calculation of times for binning when gates don't fill the interpulse time, also add calculation for numPulses
- Verify that exponential fitting is actually being done correctly (maybe need to normalize and change function)
- Determine cutoff values for variance (when to discard the beta calculated in a pixel)
'''

filename = '240529_SPAD-QD-20MHz-5f-1000g-5ns-18ps-18ps-150uW.tif'
freq = 20 # Mhz
gatelen = 0.018 # ns
gatestep = 0.018 # ns
offset = 18 # ps
tIRF = 2 # in frames
numPulses = 5 # can calculate this

# read whole tiff image
image = tf.TiffFile(filename)
image = image.asarray()
numFrames, xlen, ylen = image.shape

# calculate length of each exponential trace to fit
gateseq = (1/(freq*1e6))/(gatestep)
tracelen = gateseq - tIRF


#create timeseries for histogram (nanoseconds)
times = np.zeros((int(len(image)/numPulses)-tIRF))
for i in range(1, int(len(image)/numPulses)):
    if (tIRF + i > (len(image)/numPulses)):
        break
    times[i-1] += (offset*1e-12) + (tIRF + i)*(gatestep*1e-9) - (gatelen*1e-9) # could instead choose middle of gate, need more info about how photons are dumped into gates

# helper function for exponential fitting with scipy.optimize.curve_fit
def fitFunc(t, a, b):
    return a*np.exp(-b*t)

# create an exponential fit for each pixel and assign beta value to a new array to be plotted
betas = np.zeros((xlen, ylen))
err = np.zeros((xlen, ylen))
for i in range(xlen):
    for j in range(ylen):
        counts = np.zeros((len(times)))
        for p in range(numPulses):
            counts += image[(tIRF + 10*p):(10 + 10*p), i, j]
        params, cov = curve_fit(fitFunc, times, counts)
        betas[i, j] = 1/params[1]
        if (abs(betas[i,j]) > 1 or betas[i,j] < 0):
            betas[i,j] = 0
        err[i,j] = cov[1, 1]    


plt.imshow(betas, cmap='hot', interpolation='nearest')
plt.colorbar(label='Decay Rate')
plt.title('Heat map of decay rates accross pixels')
plt.show()
