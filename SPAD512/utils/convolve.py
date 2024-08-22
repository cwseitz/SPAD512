import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.signal import convolve

x = np.linspace(0, 20, 1000)  
uniform = np.where((x >= 5) & (x <= 10), 1/(7.5 - 2.5), 0)  
widths = [0.1, 0.5, 1, 1.5, 2]

plt.figure(figsize=(8, 6))
plt.plot(x, uniform, label='Uniform gate', color='black')

for width in widths:
    gaussian = norm.pdf(x, loc=10, scale=width)  
    convolved = convolve(uniform, gaussian, mode='same') * (x[1] - x[0])  
    plt.plot(x, convolved, label=f'{width} ns width')

plt.xlabel('x')
plt.ylabel('Amplitude')
plt.title('Convolutions of Uniform gates with varying IRF widths, mean = 10 ns')
plt.legend()
plt.show()
