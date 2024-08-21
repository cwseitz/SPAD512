from skimage.io import imread
import matplotlib.pyplot as plt
import numpy as np
import re

fig, ax = plt.subplots()
files = ["C:\\Users\\ishaa\\Documents\\FLIM\\240821\\acq00000_240819_SPAD-10MHz-1f-1000g-10000us-5ns-18ps-18ps-300uW.tif",
         "C:\\Users\\ishaa\\Documents\\FLIM\\240821\\acq00005_240819_SPAD-10MHz-1f-1000g-10000us-10ns-18ps-18ps-300uW.tif",
         "C:\\Users\\ishaa\\Documents\\FLIM\\240821\\acq00003_240819_SPAD-10MHz-1f-1000g-10000us-8ns-18ps-18ps-300uW.tif",
         "C:\\Users\\ishaa\\Documents\\FLIM\\240821\\acq00002_240819_SPAD-10MHz-1f-1000g-10000us-7ns-18ps-18ps-300uW.tif",
         "C:\\Users\\ishaa\\Documents\\FLIM\\240821\\acq00001_240819_SPAD-10MHz-1f-1000g-10000us-6ns-18ps-18ps-300uW.tif",
         "C:\\Users\\ishaa\\Documents\\FLIM\\240821\\acq00004_240819_SPAD-10MHz-1f-1000g-10000us-9ns-18ps-18ps-300uW.tif"]
files.sort()
interval = 0.018  

for i, file in enumerate(files):
    stack = imread(file)
    times = np.arange(0, stack.shape[0] * interval, interval)[:stack.shape[0]] 
    # counts = np.mean(stack, axis=(1,2))
    counts = stack[:, 289, 258]
    width = re.search(r'(\d+)ns', file).group(1)

    # rising edge plotting 
    peak = np.argmax(np.diff(counts))
    start = max(0, peak - int(1 / interval))  
    end = peak + int(1 / interval)  

    # ax.plot(times[start:end] - times[peak], counts[start:end], label=f'{width} ns')
    ax.plot(times, counts, label=f'{width} ns')

ax.set_xlabel('Time, ns')
ax.set_ylabel('Counts')
ax.legend()
plt.title('IRF rising edge comparisons')
plt.show()
