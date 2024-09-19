import numpy as np
import matplotlib.pyplot as plt

time = np.linspace(0, 100, 1000)
intensity = np.exp(-time / 20)

regs = [(5, 15), (20, 30), (35, 45), (50, 60)]
colors = ['red', 'yellow', 'green', 'blue']

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(time, intensity, label='Exponential Decay')

for i, ((start, end), color) in enumerate(zip(regs, colors), start=1):
    mask = (time >= start) & (time <= end)
    fill_color = color
    ax.fill_between(time[mask], intensity[mask], facecolor=fill_color, edgecolor=color, alpha=0.3)

ax.set_xlabel('Time (ns)')
ax.set_ylabel('Intensity (a.u.)')
ax.set_title('Rapid Lifetime Determination')
ax.set_xlim(0, 100)
ax.set_ylim(0, 1)
ax.legend()
ax.grid(False)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.show()
