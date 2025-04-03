import numpy as np
import matplotlib.pyplot as plt

data_low = np.load(r"C:\Users\ishaa\Documents\FLIM\figure_remaking\figure1_low.npz")
data_high = np.load(r"C:\Users\ishaa\Documents\FLIM\figure_remaking\figure1_high.npz")
times = data_low['times']
raw_low = data_low['raw']
bit_low = data_low['bitted']
raw_high = data_high['raw']
bit_high = data_high['bitted']

fig, ax = plt.subplots(1, 2, figsize=(10,5))
ax[0].plot(times, raw_low/np.max(raw_low), color='black', label='No Bit-Depth')
ax[0].plot(times, bit_low/np.max(bit_low), color='red', label='8-bit')
ax[0].set_xlabel('Time (s)', fontsize=14)
ax[0].set_ylabel('Norm. Counts', fontsize=14)
ax[0].set_xticks(ticks=[0, 25, 50, 75], labels=[0, 25, 50, 75], fontsize=12)
ax[0].set_yticks(ticks=[0, 0.5, 1], labels=[0, 0.5, 1], fontsize=12)
ax[0].legend(fontsize=14)


ax[1].plot(times, raw_high/np.max(raw_high), color='black', label='No Bit-Depth')
ax[1].plot(times, bit_high/np.max(bit_high), color='red', label='8-bit')
ax[1].set_xlabel('Time (s)', fontsize=14)
ax[1].set_ylabel('Norm. Counts', fontsize=14)
ax[1].set_xticks(ticks=[0, 25, 50, 75], labels=[0, 25, 50, 75], fontsize=12)
ax[1].set_yticks(ticks=[0, 0.5, 1], labels=[0, 0.5, 1], fontsize=12)
ax[1].legend(fontsize=14)

plt.tight_layout()
plt.show()