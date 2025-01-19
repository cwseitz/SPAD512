import numpy as np
import matplotlib.pyplot as plt


def spad512(x, Imax):
    return -Imax * np.log(1 - x / Imax)

x = np.linspace(1, 254, 10000)
Imax = 255
corr = Imax * (spad512(x, Imax) / np.max(spad512(x, Imax)))

edges = [0, 50, 100, 150, 200, 255] 

plt.figure(figsize=(10, 6))
for i in range(len(edges) - 1):
    if i == len(edges) - 2:
        mask = (corr >= edges[i]) & (corr <= edges[i + 1])
    else:
        mask = (corr >= edges[i]) & (corr < edges[i + 1])
    plt.plot(corr[mask], x[mask], label=f'{edges[i]} - {edges[i + 1]}')

plt.xlabel('Corrected Counts')
plt.ylabel('Counts Recorded by SPAD')
plt.grid(True)
plt.legend(title="X Ranges", loc='lower right')  
plt.show()
