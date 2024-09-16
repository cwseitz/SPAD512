import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import pickle  # For pickling

# Set global font sizes
plt.rcParams.update({'font.size': 8})  # Adjust this value as needed

@staticmethod
def plot_lifetimes(mean_img, std_img, param1s, param2s, taus, filename, show=True, pickle_fig=False):
    ntau, nx, ny = mean_img.shape
    param2s = np.array(param2s)/1000

    # fig, ax = plt.subplots(ntau, 2, figsize=(12, 6 if ntau == 1 else 12))
    fig, ax = plt.subplots(ntau, 2, figsize=(6, 6))
    ax = ax.reshape(-1, 2)

    for i in range(ntau):
        mean_lower = min(max(taus[i] - 5 * (i+1), int(np.min(mean_img[i]))), taus[i] - 1)
        mean_upper = max(min(taus[i] + 5 * (i+1), int(np.max(mean_img[i]) + 1)), taus[i] + 1)
        mean_norm = mcolors.TwoSlopeNorm(vmin=mean_lower, vcenter=taus[i], vmax=mean_upper)
        std_norm = mcolors.Normalize(vmin=0, vmax=10)
        
        plot_panel(ax[i, 0], mean_img[i], f'Lifetimes (tau {i+1})', 'Means, ns', param1s, param2s, cmap='seismic', norm=mean_norm)
        plot_panel(ax[i, 1], std_img[i], f'Std Devs (tau {i+1})', 'St Devs, ns', param1s, param2s, cmap='plasma', norm=std_norm)

    plt.tight_layout()
    plt.savefig(filename + '_fit_results', bbox_inches='tight')
    print(f'Figure saved as {filename + "_fit_results"}')

    with open(filename + '_fit_results.pkl', 'wb') as f:
        pickle.dump(fig, f)

    if show:
        plt.show()

def plot_fvals(fval_img, param1s, param2s, filename, show=True, pickle_fig=False):
    fig, ax = plt.subplots()
    norm = mcolors.Normalize(vmin=0, vmax=5)
    param1s = np.asarray(param1s) * 1e-3
    param2s = np.asarray(param2s) * 1e-3
    fval_img = np.log10(np.abs(fval_img))

    plot_panel(ax, fval_img, f'Simulation accross integration time and step sizes for bi-exponential RLD', 'F\'-values, log scale', param1s, param2s, norm=norm)
    plt.tight_layout()
    plt.savefig(filename + '_results', bbox_inches='tight')

    with open(filename + '_results.pkl', 'wb') as f:
        pickle.dump(fig, f)

    if show:
        plt.show()

def plot_panel(ax, img, title, cbar_label, yticks, xticks, cmap='plasma', norm=None):
    nx, ny = img.shape
    cax = ax.imshow(img, cmap=cmap, norm=norm)
    
    cbar = plt.colorbar(cax, ax=ax, shrink=0.8)
    cbar.set_label(cbar_label, fontsize=8)  
    vmin, vmax = cax.get_clim()  # Get the min and max values from the color scale
    if isinstance(norm, mcolors.TwoSlopeNorm):
        vcenter = norm.vcenter
        num_ticks = 5  
        ticks_below = np.linspace(vmin, vcenter, num_ticks//2 + 1, endpoint=True)
        ticks_above = np.linspace(vcenter, vmax, num_ticks//2 + 1, endpoint=True)[1:] 
        cbar_ticks = np.concatenate([ticks_below, ticks_above])
        cbar.set_ticks(cbar_ticks)
    else:
        cbar_ticks = np.linspace(vmin, vmax, 5) 
        cbar.set_ticks(cbar_ticks)
    
    ax.set_title(title, fontsize=10)  # Set smaller title font size
    ax.set_xlabel('Integration time (ms)', fontsize=8)  # Adjust x-axis label font size
    ax.set_ylabel('Bit-depth', fontsize=8)  # Adjust y-axis label font size
    ax.set_yticks(np.linspace(0, nx, num=nx, endpoint=False))
    ax.set_yticklabels(np.round(yticks, 2), fontsize=8)  # Adjust y-tick label font size
    ax.set_xticks(np.linspace(0, ny, num=ny, endpoint=False))
    ax.set_xticklabels(np.round(xticks, 2), fontsize=8)  # Adjust x-tick label font size
    plt.setp(ax.get_xticklabels(), rotation=45, fontsize=8)  # Rotate and adjust font size of x-tick labels
