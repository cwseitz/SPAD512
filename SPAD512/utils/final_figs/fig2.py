import numpy as np
import matplotlib.pyplot as plt

def bi(x, A, lam1, B, lam2):
    y = B*np.exp(-x*lam1) + (1-B)*np.exp(-x*lam2)
    return A*y
    


'''top left'''
# data = np.load(r'C:\Users\ishaa\Documents\FLIM\figure_remaking\figure2_10ms_corrected-trace.npz')
# data2 = np.load(r'C:\Users\ishaa\Documents\FLIM\figure_remaking\figure2_10ms_uncorrected-trace.npz')

# times = data['times']
# xdat = np.linspace(np.min(times), np.max(times), 1000)
# ydat_corr = data['data']
# params_corr = data['params']
# ydat_uncorr = data2['data']
# params_uncorr = data2['params']

# fig = plt.figure(figsize=(10,5))
# plt.plot(times, ydat_corr/np.max(ydat_corr), 'o', markersize=3, color='darkgreen')
# plt.plot(xdat, bi(xdat, *params_corr)/np.max(ydat_corr), ls='--', color='darkgreen', label='Corrected')
# plt.plot(times, ydat_uncorr/np.max(ydat_uncorr), 'o', markersize=3, color='red')
# plt.plot(xdat, bi(xdat, *params_uncorr)/np.max(ydat_uncorr), ls='--', color='red', label='Uncorrected')
# plt.legend(fontsize=12)
# plt.xlabel('Time (ns)', fontsize=14)
# plt.xticks(ticks=[0, 20, 40, 60, 80], labels=[0, 20, 40, 60, 80], fontsize=12)
# plt.ylabel('Norm. Counts', fontsize=14)
# plt.yticks(ticks=[0, 0.25, 0.5, 0.75, 1], labels=[0, 0.25, 0.5, 0.75, 1], fontsize=12)
# plt.show()



'''middle left'''
# data = np.load(r'C:\Users\ishaa\Documents\FLIM\figure_remaking\figure2_10ms_corrected-hist.npz')
# tau1 = data['tau1'].flatten()
# tau2 = data['tau2'].flatten()

# fig, ax = plt.subplots(1, 2, figsize=(10,5))
# ax[0].hist(tau2, bins=20, density=True, label=f'Mean = {np.mean(tau2):.2f} ns)', alpha=0.5, color='darkgreen')
# ax[0].axvline(5, ls='--', color='k')
# ax[0].set_xlabel('Lifetimes (ns)', fontsize=14)
# ax[0].set_ylabel('Probability Density', fontsize=14)
# ax[0].set_xticks(ticks=[4, 4.5, 5, 5.5, 6], labels=[4, 4.5, 5, 5.5, 6], fontsize=12)
# ax[0].set_yticks(ticks=[0, 0.5, 1, 1.5], labels=[0, 0.5, 1, 1.5], fontsize=12)
# ax[0].legend(fontsize=12)

# ax[1].hist(tau1, bins=20, density=True, label=f'Mean = {np.mean(tau1):.2f} ns)', alpha=0.5, color='darkgreen')
# ax[1].axvline(20, ls='--', color='k')
# ax[1].set_xlabel('Lifetimes (ns)', fontsize=14)
# ax[1].set_xticks(ticks=[18, 19, 20, 21, 22], labels=[18, 19, 20, 21, 22], fontsize=12)
# ax[1].set_yticks(ticks=[0, 0.4, 0.8], labels=[0, 0.4, 0.8], fontsize=12)
# ax[1].legend(fontsize=12)

# plt.tight_layout()
# plt.show()



'''bottom left'''
# data = np.load(r'C:\Users\ishaa\Documents\FLIM\figure_remaking\figure2_10ms_uncorrected-hist.npz')
# tau1 = data['tau1'].flatten()
# tau2 = data['tau2'].flatten()

# fig, ax = plt.subplots(1, 2, figsize=(10,5))
# ax[0].hist(tau2, bins=20, density=True, label=f'Mean = {np.mean(tau2):.2f} ns)', alpha=0.5, color='red')
# ax[0].set_xlabel('Lifetimes (ns)', fontsize=14)
# ax[0].set_ylabel('Probability Density', fontsize=14)
# ax[0].set_xticks(ticks=[0, 0.0005, 0.001, 0.0015, 0.002, 0.0025], labels=[0, 0.0005, 0.001, 0.0015, 0.002, 0.0025], fontsize=12)
# ax[0].set_yticks(ticks=[0, 750, 1500, 2250, 3000], labels=[0, 750, 1500, 2250, 3000], fontsize=12)
# ax[0].legend(fontsize=12)

# ax[1].hist(tau1, bins=20, density=True, label=f'Mean = {np.mean(tau1):.2f} ns)', alpha=0.5, color='red')
# ax[1].set_xlabel('Lifetimes (ns)', fontsize=14)
# ax[1].set_xticks(ticks=[25.5,25.6,25.7,25.8,25.9], labels=[25.5,25.6,25.7,25.8,25.9], fontsize=12)
# ax[1].set_yticks(ticks=[0,2,4,6,8], labels=[0,2,4,6,8], fontsize=12)
# ax[1].legend(fontsize=12)

# plt.tight_layout()
# plt.show()



'''top middle'''
# data = np.load(r'C:\Users\ishaa\Documents\FLIM\figure_remaking\figure2_500us_corrected-trace.npz')
# data2 = np.load(r'C:\Users\ishaa\Documents\FLIM\figure_remaking\figure2_500us_uncorrected-trace.npz')

# times = data['times']
# xdat = np.linspace(np.min(times), np.max(times), 1000)
# ydat_corr = data['data']
# params_corr = data['params']
# ydat_uncorr = data2['data']
# params_uncorr = data2['params']

# fig = plt.figure(figsize=(10,5))
# plt.plot(times, ydat_corr/np.max(ydat_corr), 'o', markersize=3, color='darkgreen')
# plt.plot(xdat, bi(xdat, *params_corr)/np.max(ydat_corr), ls='--', color='darkgreen', label='Corrected')
# plt.plot(times, ydat_uncorr/np.max(ydat_uncorr), 'o', markersize=3, color='red')
# plt.plot(xdat, bi(xdat, *params_uncorr)/np.max(ydat_uncorr), ls='--', color='red', label='Uncorrected')
# plt.legend(fontsize=12)
# plt.xlabel('Time (ns)', fontsize=14)
# plt.xticks(ticks=[0, 20, 40, 60, 80], labels=[0, 20, 40, 60, 80], fontsize=12)
# plt.ylabel('Norm. Counts', fontsize=14)
# plt.yticks(ticks=[0, 0.25, 0.5, 0.75, 1], labels=[0, 0.25, 0.5, 0.75, 1], fontsize=12)
# plt.show()



'''middle middle'''
# data = np.load(r'C:\Users\ishaa\Documents\FLIM\figure_remaking\figure2_500us_corrected-hist.npz')
# tau1 = data['tau1'].flatten()
# tau2 = data['tau2'].flatten()

# fig, ax = plt.subplots(1, 2, figsize=(10,5))
# ax[0].hist(tau2, bins=20, density=True, label=f'Mean = {np.mean(tau2):.2f} ns)', alpha=0.5, color='darkgreen')
# ax[0].axvline(5, ls='--', color='k')
# ax[0].set_xlabel('Lifetimes (ns)', fontsize=14)
# ax[0].set_ylabel('Probability Density', fontsize=14)
# ax[0].set_xticks(ticks=[4, 4.5, 5, 5.5, 6], labels=[4, 4.5, 5, 5.5, 6], fontsize=12)
# ax[0].set_yticks(ticks=[0, 0.5, 1, 1.5], labels=[0, 0.5, 1, 1.5], fontsize=12)
# ax[0].legend(fontsize=12)

# ax[1].hist(tau1, bins=20, density=True, label=f'Mean = {np.mean(tau1):.2f} ns)', alpha=0.5, color='darkgreen')
# ax[1].axvline(20, ls='--', color='k')
# ax[1].set_xlabel('Lifetimes (ns)', fontsize=14)
# ax[1].set_xticks(ticks=[18, 19, 20, 21, 22], labels=[18, 19, 20, 21, 22], fontsize=12)
# ax[1].set_yticks(ticks=[0, 0.4, 0.8], labels=[0, 0.4, 0.8], fontsize=12)
# ax[1].legend(fontsize=12)

# plt.tight_layout()
# plt.show()



'''bottom middle'''
# data = np.load(r'C:\Users\ishaa\Documents\FLIM\figure_remaking\figure2_500us_uncorrected-hist.npz')
# tau1 = data['tau1'].flatten()
# tau2 = data['tau2'].flatten()

# fig, ax = plt.subplots(1, 2, figsize=(10,5))
# ax[0].hist(tau2, bins=20, density=True, label=f'Mean = {np.mean(tau2):.2f} ns)', alpha=0.5, color='red')
# ax[0].axvline(5, ls='--', color='k')
# ax[0].set_xlabel('Lifetimes (ns)', fontsize=14)
# ax[0].set_ylabel('Probability Density', fontsize=14)
# ax[0].set_xticks(ticks=[4.5, 5, 5.5, 6], labels=[4.5, 5, 5.5, 6], fontsize=12)
# ax[0].set_yticks(ticks=[0, 0.5, 1, 1.5], labels=[0, 0.5, 1, 1.5], fontsize=12)
# ax[0].legend(fontsize=12)

# ax[1].hist(tau1, bins=20, density=True, label=f'Mean = {np.mean(tau1):.2f} ns)', alpha=0.5, color='red')
# ax[1].axvline(20, ls='--', color='k')
# ax[1].set_xlabel('Lifetimes (ns)', fontsize=14)
# ax[1].set_xticks(ticks=[18, 19, 20, 21, 22], labels=[18, 19, 20, 21, 22], fontsize=12)
# ax[1].set_yticks(ticks=[0, 0.4, 0.8], labels=[0, 0.4, 0.8], fontsize=12)
# ax[1].legend(fontsize=12)

# plt.tight_layout()
# plt.show()



'''right'''
def spad512(x, Imax):
    return -Imax * np.log(1 - x / Imax)

x = np.linspace(1, 254, 10000)
Imax = 255
corr = Imax * (spad512(x, Imax) / np.max(spad512(x, Imax)))
edges = [0, 50, 100, 150, 200, 255]

output_ranges = []
bar_colors = []  # Store colors for bar chart

# Main curve plot with segmented colors
plt.figure(figsize=(10, 6))
for i in range(len(edges) - 1):
    if i == len(edges) - 2:
        mask = (corr >= edges[i]) & (corr <= edges[i + 1])
    else:
        mask = (corr >= edges[i]) & (corr < edges[i + 1])
    
    line, = plt.plot(corr[mask], x[mask], label=f'{edges[i]} - {edges[i + 1]}', linewidth=3)
    color = line.get_color()
    bar_colors.append(color)  # Save for second plot

    x_end = corr[mask][-1]
    y_start = x[mask][0]
    y_end = x[mask][-1]

    output_range = y_end - y_start
    output_ranges.append(output_range)
    
    plt.plot([x_end, x_end], [0, y_end], linestyle='--', color=color, linewidth=1)
    plt.plot([0, x_end], [y_end, y_end], linestyle='--', color=color, linewidth=1)

plt.xlabel('Corrected Counts', fontsize=14)
plt.ylabel('Counts Recorded by SPAD', fontsize=14)
plt.xlim(0, 255)
plt.ylim(0, 255)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 5))
bin_labels = [f'{edges[i]}-{edges[i+1]}' for i in range(len(edges)-1)]
normalized_ranges = output_ranges / np.max(output_ranges)
plt.bar(bin_labels, normalized_ranges, color=bar_colors)
plt.xlabel('True Count Range (cts)', fontsize=14)
plt.ylabel('Normalized Recorded Count Range', fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.tight_layout()
plt.show()