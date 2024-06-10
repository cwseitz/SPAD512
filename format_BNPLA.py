import numpy as np
from scipy.io import savemat
from skimage.io import imread 
import random

'''
Script for adjusting data collected in a .tif for use in the matlab package BNP-LA by Mohamadreza Fazel.
To-Do: 
- Remake to localize relevant coordinates and iteratively make for MATLAB arrays (if working with dot samples)
- Import MATLAB package and run BNP-LA analysis on arrays as a widefield implementation
'''
x = 147 # x coordinate of interest
y = 215 # y coordinate of interest
bin_width = 0.09 # spacing of bins in nanoseconds (NOT SPAD512 gate width)
start = 2.5 + 0.018 # start point for bins (taken as half of gate width plus initial offset)
file = '240604/240604_10ms_adjusted.tif'

image = imread(file)
length, xdim, ydim = image.shape
raw = image[:, x, y]
data = []
for i, count in enumerate(raw):
    for j in range(int(count)):
        data.append(start + bin_width * i)

random.shuffle(data) # needed so that when not using all photons, data isn't skewed to taking early times only

data = np.array(data).reshape(1, -1)
mat_dict = {'Dt': data}
savemat('BNPtest.mat', mat_dict)
print(data)
print("Data saved to BNPtest.mat")
