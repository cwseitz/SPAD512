import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import json

'''chatgpt code because i just needed it replotted quickly'''

config_path = "C:\\Users\\ishaa\\Documents\\FLIM\\SPAD512\\SPAD512\\mains\\run_sim_full.json"

def load_config(config_path):
    with open(config_path) as f:
        config = json.load(f)
    param1s = config['integ']  # Integration times
    param2s = config['step']   # Step sizes
    return param1s, param2s

def plot_panel(ax, img, title, cbar_label, yticks, xticks, norm=None):
    nx, ny = img.shape
    cax = ax.imshow(img, cmap='plasma', norm=norm)
    cbar = plt.colorbar(cax, ax=ax, shrink=0.8)
    cbar.set_label(cbar_label)
    ax.set_title(title)
    ax.set_xlabel('Step size (ns)')
    ax.set_ylabel('Integration time (us)')
    ax.set_yticks(np.linspace(0, nx - 1, num=nx, endpoint=True))
    ax.set_yticklabels(np.round(yticks, 2))
    ax.set_xticks(np.linspace(0, ny - 1, num=ny, endpoint=True))
    ax.set_xticklabels(np.round(xticks, 2))
    plt.setp(ax.get_xticklabels(), rotation=45)

def plot_fvals_side_by_side(rld_filename, nnls_filename, config_path, show=True):
    param1s, param2s = load_config(config_path)

    rld_data = np.load(rld_filename)
    rld_fvals = rld_data['f_vals'].astype(float)

    nnls_data = np.load(nnls_filename)
    nnls_fvals = nnls_data['f_vals'].astype(float)

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    norm = mcolors.Normalize(vmin=0, vmax=5)
    param1s = np.asarray(param1s) * 1e-3  # Convert to us
    param2s = np.asarray(param2s) * 1e-3  # Convert to ns

    rld_fvals_log = np.log10(np.abs(rld_fvals))
    plot_panel(axes[0], rld_fvals_log, 'A) RLD', 'F\'-values, log scale', param1s, param2s, norm=norm)

    nnls_fvals_log = np.log10(np.abs(nnls_fvals))
    plot_panel(axes[1], nnls_fvals_log, 'B) NNLS', 'F\'-values, log scale', param1s, param2s, norm=norm)

    plt.tight_layout()
    plt.savefig('combined_rld_nnls_results.png', bbox_inches='tight')

    if show:
        plt.show()

rld_filename = "C:\\Users\\ishaa\\Documents\\FLIM\\ManFigs\\rld_fvals_integ_step_results.npz"
nnls_filename = "C:\\Users\\ishaa\\Documents\\FLIM\\ManFigs\\nnls_fvals_integ_step_results.npz"

plot_fvals_side_by_side(rld_filename, nnls_filename, config_path, show=True)