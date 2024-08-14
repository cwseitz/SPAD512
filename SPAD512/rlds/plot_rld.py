import numpy as np
import matplotlib.pyplot as plt

time = np.linspace(0, 10, 100)
intensity = np.exp(-time / 2)

regs = [(1, 2), (5, 6)]
cols = ['yellow', 'cyan']

plt.plot(time, intensity, label='Exponential Decay')

for (start, end), color in zip(regs, cols):
    mask = (time >= start) & (time <= end)
    plt.fill_between(time[mask], intensity[mask], color=color, alpha=0.3)


plt.xlabel('Time (ns)')
plt.ylabel('Intensity (a.u.)')
plt.legend()
plt.title('Rapid Lifetime Determination')
plt.show()