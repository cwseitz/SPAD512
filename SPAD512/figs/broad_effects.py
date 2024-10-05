import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

plt.rcParams.update({'font.size': 8})

def plot_panel(ax, img, title, cbar_label, yticks, xticks, cmap='seismic', norm=None):
    nx, ny = img.shape
    cax = ax.imshow(img, cmap=cmap, norm=norm)
    cbar = plt.colorbar(cax, ax=ax, shrink=0.8)
    cbar.set_label(cbar_label, fontsize=8)
    
    ax.set_title(title, fontsize=10)
    ax.set_xlabel('Integration time (ms)', fontsize=8)
    ax.set_ylabel('Step size (ns)', fontsize=8)
    
    ax.set_yticks(np.arange(len(yticks)))
    ax.set_yticklabels(np.round(yticks, 2), fontsize=8)
    
    ax.set_xticks(np.arange(len(xticks)))
    ax.set_xticklabels(np.round(xticks, 2), fontsize=8)
    
    plt.setp(ax.get_xticklabels(), rotation=45, fontsize=8)

def plot_lifetime(ax, img, tau, param1s, param2s, title):
    mean_lower = min(max(tau - 5, np.min(img)), tau - 1)
    mean_upper = max(min(tau + 5, np.max(img) + 1), tau + 1)
    mean_norm = mcolors.TwoSlopeNorm(vmin=mean_lower, vcenter=tau, vmax=mean_upper)
    plot_panel(ax, img, title, 'Means, ns', param1s, param2s, cmap='seismic', norm=mean_norm)

def load_data(filename):
    results = np.load(filename)
    means = results['means'].astype(float)
    param1s = np.array([500, 750, 1000, 2500, 3750, 5000, 6250, 7500, 8750, 10000]).astype(float) 
    param2s = np.array([500, 750, 1000, 2500, 3750, 5000, 6250, 7500, 8750, 10000]).astype(float)  
    param1s /= 1000
    param2s /= 1000
    lifetimes = np.array([20, 5]).astype(float)
    
    return means, param1s, param2s, lifetimes

c8_file = r'C:\Users\ishaa\Documents\FLIM\ManFigs\broad_effects\8bit_steps_integs_corrected_results.npz'
c6_file = r'C:\Users\ishaa\Documents\FLIM\ManFigs\broad_effects\6bit_steps_integs_corrected_results.npz'
u8_file = r'C:\Users\ishaa\Documents\FLIM\ManFigs\broad_effects\8bit_steps_integs_uncorrected_results.npz'
u6_file = r'C:\Users\ishaa\Documents\FLIM\ManFigs\broad_effects\6bit_steps_integs_uncorrected_results.npz'

means_c8, p1s_c8, p2s_c8, taus_c8 = load_data(c8_file)
means_c6, p1s_c6, p2s_c6, taus_c6 = load_data(c6_file)
means_u8, p1s_u8, p2s_u8, taus_u8 = load_data(u8_file)
means_u6, p1s_u6, p2s_u6, taus_u6 = load_data(u6_file)

fig, axs = plt.subplots(2, 4, figsize=(12, 6))

plot_lifetime(axs[0, 0], means_c8[0], taus_c8[0], p1s_c8, p2s_c8, 'Corrected 8bit Short')
plot_lifetime(axs[0, 1], means_c8[1], taus_c8[1], p1s_c8, p2s_c8, 'Corrected 8bit Long')
plot_lifetime(axs[0, 2], means_c6[0], taus_c6[0], p1s_c6, p2s_c6, 'Corrected 6bit Short')
plot_lifetime(axs[0, 3], means_c6[1], taus_c6[1], p1s_c6, p2s_c6, 'Corrected 6bit Long')

plot_lifetime(axs[1, 0], means_u8[0], taus_u8[0], p1s_u8, p2s_u8, 'Uncorrected 8bit Short')
plot_lifetime(axs[1, 1], means_u8[1], taus_u8[1], p1s_u8, p2s_u8, 'Uncorrected 8bit Long')
plot_lifetime(axs[1, 2], means_u6[0], taus_u6[0], p1s_u6, p2s_u6, 'Uncorrected 6bit Short')
plot_lifetime(axs[1, 3], means_u6[1], taus_u6[1], p1s_u6, p2s_u6, 'Uncorrected 6bit Long')

plt.tight_layout()
plt.show()
