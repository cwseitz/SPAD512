import numpy as np
import matplotlib.pyplot as plt

# constant params
usb3_rate = 300 # MBps
bin_readout = 10 # microseconds
image_size = 512 # pixels



# def cal_deadtime(numframes, numsteps, bits, t_r):
#     F = numframes
#     G = numsteps
#     N = bits
    
#     acq_dead = numsteps * (image_size**2) * bits * (1.25e-7) / usb3_rate # 1.25e-7 is conversion from bits to megabytes
#     bin_dead = numsteps * (2**bits - 1) * t_r * 1e-6 # convert from microseconds to seconds
#     return numframes * (acq_dead + bin_dead)

# print(cal_deadtime(50, 50, 8, 10))
    

data = np.linspace(0, 254, 500)
corrected = -255*np.log(1-(data/255))

plt.plot(corrected, data, '--b')
plt.xlabel('True number of counts')
plt.ylabel('Recorded counts at detector')
plt.grid(True)
plt.show()