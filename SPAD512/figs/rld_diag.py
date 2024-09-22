import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid")

time = np.linspace(0, 100, 1000)
cts = np.exp(-time / 20)
regs = [(5, 15), (20, 30), (35, 45), (50, 60)]
cols = ['red', 'yellow', 'green', 'blue']

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(time, cts, label='Exponential Decay')


for i, ((start, end), color) in enumerate(zip(regs, cols), start=1):
    mask = (time >= start) & (time <= end)
    ax.fill_between(time[mask], cts[mask], facecolor=color, edgecolor=color, alpha=0.3)

ax.set_xlabel('Time (ns)', fontsize=12)
ax.set_ylabel('Intensity', fontsize=12)
ax.set_xlim(0, 100)
ax.set_ylim(0, 1)
ax.grid(False)  
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.tick_params(axis='both', which='major', labelsize=10)

plt.tight_layout()
plt.show()

