import numpy as np
import matplotlib.pyplot as plt

sat = np.load("C:\\Users\\ishaa\\Documents\\FLIM\\ManFigs\\sat_traces\\saturated_trace.npz")
unsat = np.load("C:\\Users\\ishaa\\Documents\\FLIM\\ManFigs\\sat_traces\\unsaturated_trace.npz")

sat_y= sat['y']
unsat_y = unsat['y']
times = sat['x']  # Assuming both traces have the same time axis

fig, axs = plt.subplots(1, 2, figsize=(10, 4))

axs[0].plot(times, sat_y, 'bo', markersize=3, label='Saturated Trace')
axs[0].set_title('Saturated Trace', fontsize=12)
axs[0].set_xlabel('Time (ns)')
axs[0].set_ylabel('Counts')

axs[1].plot(times, unsat_y, 'bo', markersize=3, label='Unsaturated Trace')
axs[1].set_title('Unsaturated Trace', fontsize=12)
axs[1].set_xlabel('Time (ns)')
axs[1].set_ylabel('Counts')

plt.tight_layout
plt.show()