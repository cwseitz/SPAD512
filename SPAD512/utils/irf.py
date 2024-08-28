from skimage.io import imread
import matplotlib.pyplot as plt
import numpy as np
import re
import scipy.optimize as opt

fig, ax = plt.subplots()
# files = ["C:\\Users\\ishaa\\Documents\\FLIM\\240821\\acq00000_240819_SPAD-10MHz-1f-1000g-10000us-5ns-18ps-18ps-300uW.tif",
#          "C:\\Users\\ishaa\\Documents\\FLIM\\240821\\acq00005_240819_SPAD-10MHz-1f-1000g-10000us-10ns-18ps-18ps-300uW.tif",
#          "C:\\Users\\ishaa\\Documents\\FLIM\\240821\\acq00003_240819_SPAD-10MHz-1f-1000g-10000us-8ns-18ps-18ps-300uW.tif",
#          "C:\\Users\\ishaa\\Documents\\FLIM\\240821\\acq00002_240819_SPAD-10MHz-1f-1000g-10000us-7ns-18ps-18ps-300uW.tif",
#          "C:\\Users\\ishaa\\Documents\\FLIM\\240821\\acq00001_240819_SPAD-10MHz-1f-1000g-10000us-6ns-18ps-18ps-300uW.tif",
#          "C:\\Users\\ishaa\\Documents\\FLIM\\240821\\acq00004_240819_SPAD-10MHz-1f-1000g-10000us-9ns-18ps-18ps-300uW.tif"]
files = ["C:\\Users\\ishaa\\Documents\\FLIM\\240813\\240813_SPAD-10MHz-1f-500g-10000us-5ns-100ps-18ps-1500uW.tif",
         "C:\\Users\\ishaa\\Documents\\FLIM\\240821\\acq00000_240819_SPAD-10MHz-1f-1000g-10000us-5ns-18ps-18ps-300uW.tif"]
intervals = [0.1, 0.018]
names = ['Alexa', 'Eosin']

def decay(x, A, tau):
    return A * np.exp(-x/tau)

for i, file in enumerate(files):
    stack = imread(file)
    times = np.arange(0, stack.shape[0] * intervals[i], intervals[i])[:stack.shape[0]] 
    counts = np.mean(stack, axis=(1,2))
    counts /= np.max(counts)

    # # rising edge plotting 
    # peak = np.argmax(np.diff(counts))
    # start = peak - int(2/intervals[i])
    # end = peak + int(2/intervals[i])
    # if (i==0):
    #     end = peak + int(3/intervals[i])
    # times = times[start:end]
    # counts = counts[start:end]

    ax.plot(times, counts, label=names[i])
    # ax.plot(times[start:], decay(times[start:], *popt), label='Tau: %1.3f, A: %3.3f' % tuple(popt))

ax.set_xlabel('Time, ns')
ax.set_ylabel('Counts')
ax.legend()
plt.title('Full profile comparisons')
plt.show()
