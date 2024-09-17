import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import math

def plot_tau2_histograms(npz_files, output_file, show=True):
    bit6_files = [f for f in npz_files if '6bit' in f]
    bit4_files = [f for f in npz_files if '4bit' in f]

    datasets = []
    for bit_files in [bit6_files, bit4_files]:
        for file in bit_files:
            data = np.load(file)
            if 'tau2' in data:
                tau2 = data['tau2'].flatten()
                tau2 = tau2[(tau2 != 0) & (tau2 >= 2) & (tau2 <= 8)]
            else:
                raise ValueError(f"File {file} does not contain 'tau2' entry.")

            match = re.search(r'(\d+)k', file)
            if match:
                k_raw = int(match.group(1))
                kernel_size = 2 * k_raw + 1
                k_value = kernel_size * kernel_size  
                pixels = f'{k_value} binned'
            else:
                k_value = None
                pixels = 'Unknown binned'

            datasets.append({
                'tau2': tau2,
                'file': file,
                'k_value': k_value,
                'pixels': pixels,
                'bit_depth': '6-bit' if '6bit' in file else '4-bit'
            })


    sns.set_theme(style="white")  # Changed from sns.set to sns.set_theme

    fig, ax = plt.subplots(2, 3, figsize=(15, 8), squeeze=False)
    colors = sns.color_palette("muted", n_colors=6)
    for idx, dataset in enumerate(datasets):
        row = 0 if dataset['bit_depth'] == '6-bit' else 1
        col = idx % 3

        sns.histplot(
            dataset['tau2'],
            bins=50,
            kde=True,
            stat='density', 
            color=colors[idx],
            ax=ax[row, col]
        )

        ax[row, col].set_xlim(2, 8)
        ax[row, col].set_xlabel('Smaller lifetime (ns)')
        ax[row, col].set_ylabel('Density')
        ax[row, col].spines['top'].set_visible(False)
        ax[row, col].spines['right'].set_visible(False)

        if dataset['k_value'] is not None:
            frames_needed = math.ceil(dataset['k_value'] / 49)
            ax[row, col].set_title(f"{frames_needed} frames", fontweight='bold')  # Added fontweight='bold'
        else:
            ax[row, col].set_title("Unknown frames", fontweight='bold')  # Added fontweight='bold'

    plt.tight_layout()
    plt.savefig(output_file, bbox_inches='tight')
    print(f'Figure saved as {output_file}')

    if show:
        plt.show()

npz_files = [
    r"C:\\Users\\ishaa\\Documents\\FLIM\\ManFigs\\bitstuffs\\50us_4bit_11k_fit_results.npz",
    r"C:\\Users\\ishaa\\Documents\\FLIM\\ManFigs\\bitstuffs\\50us_4bit_13k_fit_results.npz",
    r"C:\\Users\\ishaa\\Documents\\FLIM\\ManFigs\\bitstuffs\\50us_4bit_15k_fit_results.npz",
    r"C:\\Users\\ishaa\\Documents\\FLIM\\ManFigs\\bitstuffs\\500us_6bit_5k_fit_results.npz",
    r"C:\\Users\\ishaa\\Documents\\FLIM\\ManFigs\\bitstuffs\\500us_6bit_7k_fit_results.npz",
    r"C:\\Users\\ishaa\\Documents\\FLIM\\ManFigs\\bitstuffs\\500us_6bit_9k_fit_results.npz"
]

output_file = r"C:\\Users\\ishaa\\Documents\\FLIM\\ManFigs\\tau2_histograms.png"
plot_tau2_histograms(npz_files, output_file, show=True)
