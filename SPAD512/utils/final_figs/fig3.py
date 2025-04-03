import numpy as np
import matplotlib.pyplot as plt

'''left'''
data = np.load(r'C:\Users\ishaa\Documents\FLIM\figure_remaking\figure3_timeseries.npz')
integs = data['integs']
tau1s = data['tau1s']
tau1s_err = data['tau1s_err']
tau2s = data['tau2s']
tau2s_err = data['tau2s_err']

plt.figure(figsize=(8,8))
plt.errorbar(integs, tau1s, yerr=tau1s_err, capsize=3, fmt='bo')
plt.axhline(20, ls='--', color='blue', alpha=0.5)
plt.errorbar(integs, tau2s, yerr=tau2s_err, capsize=3, fmt='ro')
plt.axhline(5, ls='--', color='red', alpha=0.5)
plt.xlabel('Integration Time (ms)', fontsize=14)
plt.ylabel('Mean Lifetime (ns)', fontsize=14)
plt.xticks(ticks=[0,5,10,15,20,25], fontsize=12)
plt.yticks(ticks=[0,5,10,15,20], fontsize=12)
plt.show()



'''top, note that 7o5 files got overwritten so make sure to regenerate those if edits need to be made''' 
# fig, ax = plt.subplots(1, 3, figsize=(9,3))
# data = np.load(r"C:\Users\ishaa\Documents\FLIM\figure_remaking\figure3_7o5ms-trace.npz")
# xdat = data['x']
# ydat = data['data']
# ax[0].plot(xdat, ydat, 'ok', markersize=2)
# ax[0].axhline(255, ls='--',  color='red', linewidth=1)
# ax[0].set_xlabel('Time (ns)', fontsize=14)
# ax[0].set_xticks(ticks=[0, 25, 50, 75], labels=[0,25,50,75], fontsize=12)
# ax[0].set_ylabel('Counts (cts)', fontsize=14)
# ax[0].set_yticks(ticks=[0, 100, 200, 255], labels=[0, 100, 200, 255], fontsize=12)

# data = np.load(r"C:\Users\ishaa\Documents\FLIM\figure_remaking\figure3_7o5ms-hist.npz")
# tau1 = data['tau1'].flatten()
# tau2 = data['tau2'].flatten()

# ax[1].hist(tau2, bins=20, density=True, label=f'Mean = {np.mean(tau2):.2f} ns)', alpha=0.5, color='red')
# ax[1].axvline(5, ls='--', color='k')
# ax[1].set_xlabel('Lifetimes (ns)', fontsize=14)
# ax[1].set_ylabel('Probability Density', fontsize=14)
# ax[1].set_xticks(ticks=[4, 4.5, 5, 5.5, 6], labels=[4, 4.5, 5, 5.5, 6], fontsize=12)
# ax[1].set_yticks(ticks=[0,2, 4], labels=[0, 2, 4], fontsize=12)
# ax[1].legend(fontsize=12)

# ax[2].hist(tau1, bins=20, density=True, label=f'Mean = {np.mean(tau1):.2f} ns)', alpha=0.5, color='blue')
# ax[2].axvline(20, ls='--', color='k')
# ax[2].set_xlabel('Lifetimes (ns)', fontsize=14)
# ax[2].set_xticks(ticks=[18, 19, 20, 21, 22], labels=[18, 19, 20, 21, 22], fontsize=12)
# ax[2].set_yticks(ticks=[0, 1, 2], labels=[0, 1, 2], fontsize=12)
# ax[2].legend(fontsize=12)

# plt.tight_layout()
# plt.show()



'''middle'''
# fig, ax = plt.subplots(1, 3, figsize=(9,3))
# data = np.load(r"C:\Users\ishaa\Documents\FLIM\figure_remaking\figure3_10ms-trace.npz")
# xdat = data['x']
# ydat = data['data']
# ax[0].plot(xdat, ydat, 'ok', markersize=2)
# ax[0].axhline(255, ls='--',  color='red', linewidth=1)
# ax[0].set_xlabel('Time (ns)', fontsize=14)
# ax[0].set_xticks(ticks=[0, 25, 50, 75], labels=[0,25,50,75], fontsize=12)
# ax[0].set_ylabel('Counts (cts)', fontsize=14)
# ax[0].set_yticks(ticks=[0, 100, 200, 255], labels=[0, 100, 200, 255], fontsize=12)

# data = np.load(r"C:\Users\ishaa\Documents\FLIM\figure_remaking\figure3_10ms-hist.npz")
# tau1 = data['tau1'].flatten()
# tau2 = data['tau2'].flatten()

# ax[1].hist(tau2, bins=20, density=True, label=f'Mean = {np.mean(tau2):.2f} ns)', alpha=0.5, color='red')
# ax[1].axvline(5, ls='--', color='k')
# ax[1].set_xlabel('Lifetimes (ns)', fontsize=14)
# ax[1].set_ylabel('Probability Density', fontsize=14)
# ax[1].set_xticks(ticks=[4, 4.5, 5, 5.5, 6], labels=[4, 4.5, 5, 5.5, 6], fontsize=12)
# ax[1].set_yticks(ticks=[0,1, 2], labels=[0, 1, 2], fontsize=12)
# ax[1].legend(fontsize=12)

# ax[2].hist(tau1, bins=20, density=True, label=f'Mean = {np.mean(tau1):.2f} ns)', alpha=0.5, color='blue')
# ax[2].axvline(20, ls='--', color='k')
# ax[2].set_xlabel('Lifetimes (ns)', fontsize=14)
# ax[2].set_xticks(ticks=[18, 19, 20, 21, 22], labels=[18, 19, 20, 21, 22], fontsize=12)
# ax[2].set_yticks(ticks=[0, 0.5, 1], labels=[0, 0.5, 1], fontsize=12)
# ax[2].legend(fontsize=12)

# plt.tight_layout()
# plt.show()



'''bottom'''
# fig, ax = plt.subplots(1, 3, figsize=(9,3))
# data = np.load(r"C:\Users\ishaa\Documents\FLIM\figure_remaking\figure3_20ms-trace.npz")
# xdat = data['x']
# ydat = data['data']
# ax[0].plot(xdat, ydat, 'ok', markersize=2)
# ax[0].axhline(255, ls='--',  color='red', linewidth=1)
# ax[0].set_xlabel('Time (ns)', fontsize=14)
# ax[0].set_xticks(ticks=[0, 25, 50, 75], labels=[0,25,50,75], fontsize=12)
# ax[0].set_ylabel('Counts (cts)', fontsize=14)
# ax[0].set_yticks(ticks=[0, 100, 200, 255], labels=[0, 100, 200, 255], fontsize=12)

# data = np.load(r'C:\Users\ishaa\Documents\FLIM\figure_remaking\figure3_20ms-hist.npz')
# tau1 = data['tau1'].flatten()
# tau2 = data['tau2'].flatten()

# ax[1].hist(tau2, bins=20, density=True, label=f'Mean = {np.mean(tau2):.2f} ns)', alpha=0.5, color='red')
# ax[1].set_xlabel('Lifetimes (ns)', fontsize=14)
# ax[1].set_ylabel('Probability Density', fontsize=14)
# ax[1].set_xticks(ticks=[1, 3, 5, 7], labels=[1, 3, 5, 7], fontsize=12)
# ax[1].set_yticks(ticks=[0, 0.75, 1.5], labels=[0, 0.75, 1.5], fontsize=12)
# ax[1].legend(fontsize=12)

# ax[2].hist(tau1, bins=20, density=True, label=f'Mean = {np.mean(tau1):.2f} ns)', alpha=0.5, color='blue')
# ax[2].set_xlabel('Lifetimes (ns)', fontsize=14)
# ax[2].set_xticks(ticks=[10, 14, 18, 22, 26], labels=[10, 14, 18, 22, 26], fontsize=12)
# ax[2].set_yticks(ticks=[0, 0.2, 0.4], labels=[0, 0.2, 0.4], fontsize=12)
# ax[2].legend(fontsize=12)

# plt.tight_layout()
# plt.show()
