import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import json

filename_corr = "C:\\Users\\ishaa\\Documents\\FLIM\\ManFigs\\corr_validate\\steps_integs_corrected"
with open(filename_corr + "_mtdt.json") as f:
    config_corr = json.load(f)
data_corr = np.load(filename_corr + "_results.npz")
means_corr = data_corr['means']
stdevs_corr = data_corr['stdevs']

filename_uncorr = "C:\\Users\\ishaa\\Documents\\FLIM\\ManFigs\\corr_validate\\steps_integs_uncorrected"
with open(filename_uncorr + "_mtdt.json") as f:
    config_uncorr = json.load(f)
data_uncorr = np.load(filename_uncorr + "_results.npz")
means_uncorr = data_uncorr['means']
stdevs_uncorr = data_uncorr['stdevs']

param1s = np.array(config_corr['step']) / 1000  # Convert ps to ns
param2s = np.array(config_corr['integ']) / 1000  # Convert us to ms
taus = config_corr['lifetimes']

if taus[0] < taus[1]:
    idx_small, idx_large = 0, 1
else:
    idx_small, idx_large = 1, 0

fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(16, 8))

# panel plotter copied from plot_sim.py
def plot_panel(ax, img, title, cbar_label, yticks, xticks, cmap='plasma', norm=None):
    cax = ax.imshow(img, cmap=cmap, norm=norm)
    cbar = plt.colorbar(cax, ax=ax, shrink=0.8)
    cbar.set_label(cbar_label, fontsize=8)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel('Integration time (ms)', fontsize=8)
    ax.set_ylabel('Gate Step Size (ns)', fontsize=8)
    nx, ny = img.shape
    ax.set_xticks(np.arange(ny))
    ax.set_xticklabels(np.round(xticks, 2), fontsize=8)
    ax.set_yticks(np.arange(nx))
    ax.set_yticklabels(yticks, fontsize=8)
    plt.setp(ax.get_xticklabels(), rotation=45, fontsize=8)

# norming functions so i don't have to retype
def get_mean_norm(tau, img):
    vmin = tau * 0.5
    vmax = tau * 1.5
    return mcolors.TwoSlopeNorm(vmin=vmin, vcenter=tau, vmax=vmax)

def get_std_norm(tau):
    return mcolors.Normalize(vmin=0, vmax=tau * 0.5) 

# uncorrected
mean_img = means_uncorr[idx_small, :, :]
std_img = stdevs_uncorr[idx_small, :, :]
mean_norm = get_mean_norm(taus[idx_small], mean_img)
std_norm = get_std_norm(taus[idx_small])
plot_panel(axes[0, 0], mean_img, f'Uncorrected: Mean Tau (Smaller)', 'Means, ns', param1s, param2s, cmap='seismic', norm=mean_norm)
plot_panel(axes[0, 1], std_img, f'Uncorrected: Std Dev Tau (Smaller)', 'St Devs, ns', param1s, param2s, cmap='plasma', norm=std_norm)

mean_img = means_uncorr[idx_large, :, :]
std_img = stdevs_uncorr[idx_large, :, :]
mean_norm = get_mean_norm(taus[idx_large], mean_img)
std_norm = get_std_norm(taus[idx_large])
plot_panel(axes[0, 2], mean_img, f'Uncorrected: Mean Tau (Larger)', 'Means, ns', param1s, param2s, cmap='seismic', norm=mean_norm)
plot_panel(axes[0, 3], std_img, f'Uncorrected: Std Dev Tau (Larger)', 'St Devs, ns', param1s, param2s, cmap='plasma', norm=std_norm)

# corrected
mean_img = means_corr[idx_small, :, :]
std_img = stdevs_corr[idx_small, :, :]
mean_norm = get_mean_norm(taus[idx_small], mean_img)
std_norm = get_std_norm(taus[idx_small])
plot_panel(axes[1, 0], mean_img, f'Corrected: Mean Tau (Smaller)', 'Means, ns', param1s, param2s, cmap='seismic', norm=mean_norm)
plot_panel(axes[1, 1], std_img, f'Corrected: Std Dev Tau (Smaller)', 'St Devs, ns', param1s, param2s, cmap='plasma', norm=std_norm)

mean_img = means_corr[idx_large, :, :]
std_img = stdevs_corr[idx_large, :, :]
mean_norm = get_mean_norm(taus[idx_large], mean_img)
std_norm = get_std_norm(taus[idx_large])
plot_panel(axes[1, 2], mean_img, f'Corrected: Mean Tau (Larger)', 'Means, ns', param1s, param2s, cmap='seismic', norm=mean_norm)
plot_panel(axes[1, 3], std_img, f'Corrected: Std Dev Tau (Larger)', 'St Devs, ns', param1s, param2s, cmap='plasma', norm=std_norm)

plt.tight_layout()
plt.savefig('C:\\Users\\ishaa\\Documents\\FLIM\\ManFigs\\new\\composite.png', bbox_inches='tight')
plt.show()
