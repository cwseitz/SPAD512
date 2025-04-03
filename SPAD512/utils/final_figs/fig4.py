import numpy as np
import matplotlib.pyplot as plt


'''4 bits, 10 steps'''
# trace_data = np.load(r"C:\Users\ishaa\Documents\FLIM\figure_remaking\figure4_4bits_10steps-trace.npz")
# xdat = trace_data['x']
# ydat = trace_data['data']
# hist_data = np.load(r"C:\Users\ishaa\Documents\FLIM\figure_remaking\figure4_4bits_10steps-hist.npz")
# tau1 = hist_data['tau1'].flatten()
# tau2 = hist_data['tau2'].flatten()

# fig, ax = plt.subplots(1, 3, figsize=(12,3))
# ax[0].plot(xdat, ydat, 'ok', markersize=2)
# ax[0].axhline(15, ls='--',  color='red', linewidth=1)
# ax[0].set_xlabel('Time (ns)', fontsize=14)
# ax[0].set_xticks(ticks=[0, 25, 50, 75], labels=[0,25,50,75], fontsize=12)
# ax[0].set_ylabel('Counts (cts)', fontsize=14)
# ax[0].set_yticks(ticks=[0, 5, 10, 15], labels=[0, 5, 10, 15], fontsize=12)

# ax[1].hist(tau2, bins=20, density=True, label=f'Mean = {np.mean(tau2):.2f} ns', alpha=0.5, color='red')
# ax[1].axvline(5, ls='--', color='k')
# ax[1].set_xlabel('Lifetimes (ns)', fontsize=14)
# ax[1].set_ylabel('Probability Density', fontsize=14)
# ax[1].set_xticks(ticks=[4, 5, 6, 7], labels=[4, 5, 6, 7], fontsize=12)
# ax[1].set_yticks(ticks=[0,0.3, 0.6], labels=[0,0.3, 0.6], fontsize=12)
# ax[1].legend(fontsize=12)

# ax[2].hist(tau1, bins=20, density=True, label=f'Mean = {np.mean(tau1):.2f} ns', alpha=0.5, color='blue')
# ax[2].axvline(20, ls='--', color='k')
# ax[2].set_xlabel('Lifetimes (ns)', fontsize=14)
# ax[2].set_xticks(ticks=[18, 20, 22, 24, 26], labels=[18, 20, 22, 24, 26], fontsize=12)
# ax[2].set_yticks(ticks=[0, 0.15, 0.3], labels=[0, 0.15, 0.3], fontsize=12)
# ax[2].legend(fontsize=12)

# plt.tight_layout()
# plt.show()

'''4 bits, 100 steps'''
# trace_data = np.load(r"C:\Users\ishaa\Documents\FLIM\figure_remaking\figure4_4bits_100steps-trace.npz")
# xdat = trace_data['x']
# ydat = trace_data['data']
# hist_data = np.load(r"C:\Users\ishaa\Documents\FLIM\figure_remaking\figure4_4bits_100steps-hist.npz")
# tau1 = hist_data['tau1'].flatten()
# tau2 = hist_data['tau2'].flatten()

# fig, ax = plt.subplots(1, 3, figsize=(12,3))
# ax[0].plot(xdat, ydat, 'ok', markersize=2)
# ax[0].axhline(15, ls='--',  color='red', linewidth=1)
# ax[0].set_xlabel('Time (ns)', fontsize=14)
# ax[0].set_xticks(ticks=[0, 25, 50, 75], labels=[0,25,50,75], fontsize=12)
# ax[0].set_ylabel('Counts (cts)', fontsize=14)
# ax[0].set_yticks(ticks=[0, 5, 10, 15], labels=[0, 5, 10, 15], fontsize=12)

# ax[1].hist(tau2, bins=20, density=True, label=f'Mean = {np.mean(tau2):.2f} ns', alpha=0.5, color='red')
# ax[1].axvline(5, ls='--', color='k')
# ax[1].set_xlabel('Lifetimes (ns)', fontsize=14)
# ax[1].set_ylabel('Probability Density', fontsize=14)
# ax[1].set_xticks(ticks=[4, 5, 6], labels=[4, 5, 6], fontsize=12)
# ax[1].set_yticks(ticks=[0, 0.5, 1], labels=[0, 0.5, 1], fontsize=12)
# ax[1].legend(fontsize=12)

# ax[2].hist(tau1, bins=20, density=True, label=f'Mean = {np.mean(tau1):.2f} ns', alpha=0.5, color='blue')
# ax[2].axvline(20, ls='--', color='k')
# ax[2].set_xlabel('Lifetimes (ns)', fontsize=14)
# ax[2].set_xticks(ticks=[18, 20, 22], labels=[18, 20, 22], fontsize=12)
# ax[2].set_yticks(ticks=[0, 0.2, 0.4], labels=[0, 0.2, 0.4], fontsize=12)
# ax[2].legend(fontsize=12)

# plt.tight_layout()
# plt.show()


'''12 bits, 10 steps'''
# trace_data = np.load(r"C:\Users\ishaa\Documents\FLIM\figure_remaking\figure4_12bits_10steps-trace.npz")
# xdat = trace_data['x']
# ydat = trace_data['data']
# hist_data = np.load(r"C:\Users\ishaa\Documents\FLIM\figure_remaking\figure4_12bits_10steps-hist.npz")
# tau1 = hist_data['tau1'].flatten()
# tau2 = hist_data['tau2'].flatten()

# fig, ax = plt.subplots(1, 3, figsize=(12,3))
# ax[0].plot(xdat, ydat, 'ok', markersize=2)
# ax[0].axhline(4095, ls='--',  color='red', linewidth=1)
# ax[0].set_xlabel('Time (ns)', fontsize=14)
# ax[0].set_xticks(ticks=[0, 25, 50, 75], labels=[0,25,50,75], fontsize=12)
# ax[0].set_ylabel('Counts (cts)', fontsize=14)
# ax[0].set_yticks(ticks=[0, 1000, 2000, 3000, 4095], labels=[0, 1000, 2000, 3000, 4095], fontsize=12)

# ax[1].hist(tau2, bins=20, density=True, label=f'Mean = {np.mean(tau2):.2f} ns', alpha=0.5, color='red')
# ax[1].axvline(5, ls='--', color='k')
# ax[1].set_xlabel('Lifetimes (ns)', fontsize=14)
# ax[1].set_ylabel('Probability Density', fontsize=14)
# ax[1].set_xticks(ticks=[4.9, 5, 5.1], labels=[4.9, 5, 5.1], fontsize=12)
# ax[1].set_yticks(ticks=[0, 4, 8], labels=[0, 4, 8], fontsize=12)
# ax[1].legend(fontsize=12)

# ax[2].hist(tau1, bins=20, density=True, label=f'Mean = {np.mean(tau1):.2f} ns', alpha=0.5, color='blue')
# ax[2].axvline(20, ls='--', color='k')
# ax[2].set_xlabel('Lifetimes (ns)', fontsize=14)
# ax[2].set_xticks(ticks=[19.8, 20, 20.2], labels=[19.8, 20, 20.2], fontsize=12)
# ax[2].set_yticks(ticks=[0, 3, 6], labels=[0, 3, 6], fontsize=12)
# ax[2].legend(fontsize=12)

# plt.tight_layout()
# plt.show()


'''12 bits, 100 steps'''
trace_data = np.load(r"C:\Users\ishaa\Documents\FLIM\figure_remaking\figure4_12bits_100steps-trace.npz")
xdat = trace_data['x']
ydat = trace_data['data']
hist_data = np.load(r"C:\Users\ishaa\Documents\FLIM\figure_remaking\figure4_12bits_100steps-hist.npz")
tau1 = hist_data['tau1'].flatten()
tau2 = hist_data['tau2'].flatten()

fig, ax = plt.subplots(1, 3, figsize=(12,3))
ax[0].plot(xdat, ydat, 'ok', markersize=2)
ax[0].axhline(4095, ls='--',  color='red', linewidth=1)
ax[0].set_xlabel('Time (ns)', fontsize=14)
ax[0].set_xticks(ticks=[0, 25, 50, 75], labels=[0,25,50,75], fontsize=12)
ax[0].set_ylabel('Counts (cts)', fontsize=14)
ax[0].set_yticks(ticks=[0, 1000, 2000, 3000, 4095], labels=[0, 1000, 2000, 3000, 4095], fontsize=12)

ax[1].hist(tau2, bins=20, density=True, label=f'Mean = {np.mean(tau2):.2f} ns', alpha=0.5, color='red')
ax[1].axvline(5, ls='--', color='k')
ax[1].set_xlabel('Lifetimes (ns)', fontsize=14)
ax[1].set_ylabel('Probability Density', fontsize=14)
ax[1].set_xticks(ticks=[4.8, 4.9, 5, 5.1], labels=[4.8, 4.9, 5, 5.1], fontsize=12)
ax[1].set_yticks(ticks=[0, 3, 6], labels=[0, 3, 6], fontsize=12)
ax[1].legend(fontsize=12)

ax[2].hist(tau1, bins=20, density=True, label=f'Mean = {np.mean(tau1):.2f} ns', alpha=0.5, color='blue')
ax[2].axvline(20, ls='--', color='k')
ax[2].set_xlabel('Lifetimes (ns)', fontsize=14)
ax[2].set_xticks(ticks=[19.6, 19.8, 20, 20.2], labels=[19.6, 19.8, 20, 20.2], fontsize=12)
ax[2].set_yticks(ticks=[0, 1.5, 3], labels=[0, 1.5, 3], fontsize=12)
ax[2].legend(fontsize=12)

plt.tight_layout()
plt.show()