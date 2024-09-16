import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# Function to plot the "tau1" and "tau2" entries from the .npz file
def plot_lifetimes(npz_files, output_file, show=True):
    # Load the tau1 and tau2 data from each npz file
    tau1_data = []
    tau2_data = []
    for file in npz_files:
        data = np.load(file)
        if 'tau1' in data and 'tau2' in data:
            tau1_data.append(data['tau1'])
            tau2_data.append(data['tau2'])
        else:
            raise ValueError(f"File {file} does not contain 'tau1' or 'tau2' entries.")

    # Number of datasets and the shape of tau1/tau2 data
    n_datasets = len(tau1_data)
    nx, ny = tau1_data[0].shape  # Assuming the same shape for all

    # Create a figure with one column for each .npz file, and two rows for tau1 and tau2
    fig, ax = plt.subplots(2 * 2, n_datasets, figsize=(4 * n_datasets, 4 * 2 * 2))

    # Make sure ax is 2D even if there's only one row or one column
    if 2 == 1:
        ax = ax[np.newaxis, :]
    if n_datasets == 1:
        ax = ax[:, np.newaxis]

    # Loop through the datasets and plot tau1 and tau2
    for i, (tau1, tau2) in enumerate(zip(tau1_data, tau2_data)):
        for j in range(2):
            # Define colormap normalization for tau1 and tau2
            tau1_lower = np.min(tau1[j])
            tau1_upper = np.max(tau1[j])
            tau2_lower = np.min(tau2[j])
            tau2_upper = np.max(tau2[j])

            tau1_norm = mcolors.Normalize(vmin=tau1_lower, vmax=tau1_upper)
            tau2_norm = mcolors.Normalize(vmin=tau2_lower, vmax=tau2_upper)

            # Plot tau1 data
            cax1 = ax[2 * j, i].imshow(tau1[j], cmap='seismic', norm=tau1_norm)
            fig.colorbar(cax1, ax=ax[2 * j, i], label=f'Tau1 (file {i + 1}, tau {j + 1})')

            # Plot tau2 data
            cax2 = ax[2 * j + 1, i].imshow(tau2[j], cmap='seismic', norm=tau2_norm)
            fig.colorbar(cax2, ax=ax[2 * j + 1, i], label=f'Tau2 (file {i + 1}, tau {j + 1})')

            # Set the title for each subplot
            ax[2 * j, i].set_title(f'File {i + 1} - Tau1 (tau {j + 1})')
            ax[2 * j + 1, i].set_title(f'File {i + 1} - Tau2 (tau {j + 1})')

    # Adjust layout and save the figure
    plt.tight_layout()
    plt.savefig(output_file, bbox_inches='tight')
    print(f'Figure saved as {output_file}')

    if show:
        plt.show()

# List of .npz files
npz_files = [
    r"C:\\Users\\ishaa\\Documents\\FLIM\\ManFigs\\500us_6bit_7k_fit_results.npz",
    r"C:\\Users\\ishaa\\Documents\\FLIM\\ManFigs\\500us_6bit_3k_fit_results.npz",
    r"C:\\Users\\ishaa\\Documents\\FLIM\\ManFigs\\500us_6bit_4k_fit_results.npz"
]

# Output file name for the plot
output_file = r"C:\\Users\\ishaa\\Documents\\FLIM\\ManFigs\\lifetimes_comparison.png"

# Call the function to plot tau1 and tau2 from all .npz files
plot_lifetimes(npz_files, output_file, show=True)
