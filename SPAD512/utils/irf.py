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
files = ["C:\\Users\\ishaa\\Documents\\FLIM\\240821\\acq00000_240819_SPAD-10MHz-1f-1000g-10000us-5ns-18ps-18ps-300uW.tif"]
interval = 0.018  

def decay(x, A, tau):
    return A * np.exp(-x/tau)

for i, file in enumerate(files):
    stack = imread(file)
    times = np.arange(0, stack.shape[0] * interval, interval)[:stack.shape[0]] 
    print(times)
    # counts = np.mean(stack, axis=(1,2))
    counts = stack[:, 285, 259]
    width = re.search(r'(\d+)ns', file).group(1)

    # # rising edge plotting 
    # peak = np.argmin(np.diff(counts))
    # start = max(0, peak - int(1 / interval))  
    start = 731

    # times = times[start:]
    # counts = counts[start:]

    popt, pcov = opt.curve_fit(decay, times[start:], counts[start:], p0 = [0.018, 250])

    print(times[start:])
    print(counts[start:])

    ax.plot(times[start:], counts[start:], label=f'{width} ns')
    ax.plot(times[start:], decay(times[start:], *popt), label='Tau: %1.3f, A: %3.3f' % tuple(popt))

ax.set_xlabel('Time, ns')
ax.set_ylabel('Counts')
ax.legend()
plt.title('IRF falling edge comparisons')
plt.show()
