import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

@staticmethod
def plot_lifetimes(mean_img, std_img, param1s, param2s, taus, filename, show=True):
    ntau, nx, ny = mean_img.shape

    def plot_panel(ax, img, title, cbar_label, yticks, xticks, norm):
        cax = ax.imshow(img, cmap='seismic', norm=norm)
        cbar = plt.colorbar(cax, ax=ax, shrink=0.6)
        cbar.set_label(cbar_label)
        ax.set_title(title)
        ax.set_xlabel('Step size (ns)')
        ax.set_ylabel('Widths (ns)')
        ax.set_yticks(np.linspace(0, nx, num=nx, endpoint=False))
        ax.set_yticklabels(np.round(yticks, 2))
        ax.set_xticks(np.linspace(0, ny, num=ny, endpoint=False))
        ax.set_xticklabels(np.round(xticks, 2))
        plt.setp(ax.get_xticklabels(), rotation=45)

    fig, ax = plt.subplots(ntau, 2, figsize=(12, 6 if ntau == 1 else 12))
    ax = ax.reshape(-1, 2)

    for i in range(ntau):
        mean_lower = min(max(taus[i] - 5 * (i+1), int(np.min(mean_img[i]))), taus[i] - 1)
        mean_upper = max(min(taus[i] + 5 * (i+1), int(np.max(mean_img[i]) + 1)), taus[i] + 1)
        mean_norm = mcolors.TwoSlopeNorm(vmin=mean_lower, vcenter=taus[i], vmax=mean_upper)
        std_norm = mcolors.TwoSlopeNorm(vmin=-1, vcenter=0, vmax=10)
        
        plot_panel(ax[i, 0], mean_img[i], f'Lifetimes (tau {i+1})', 'Means, ns', param1s, param2s, mean_norm)
        plot_panel(ax[i, 1], std_img[i], f'Std Devs (tau {i+1})', 'St Devs, ns', param1s, param2s, std_norm)

    plt.tight_layout()
    plt.savefig(filename + '_fit_results', bbox_inches='tight')
    print(f'Figure saved as {filename + '_fit_results'}')

    if show:
        plt.show()
