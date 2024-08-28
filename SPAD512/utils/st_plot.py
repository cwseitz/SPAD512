import numpy as np
import matplotlib.pyplot as plt

s_bins = np.arange(0, 6) 
t_bins = np.arange(1, 6)  

px_size = 120  
t_res = 1  

comb_px = np.zeros((len(t_bins), len(s_bins)))
s_res = np.zeros_like(comb_px)
t_res_loss = np.zeros_like(comb_px)
total_loss = np.zeros_like(comb_px)

for i, t_bin in enumerate(t_bins):
    for j, s_bin in enumerate(s_bins):
        k_size = (2 * s_bin + 1) ** 2  
        comb_px[i, j] = t_bin * k_size  
        s_res[i, j] = (2 * s_bin + 1) * px_size  
        t_res_loss[i, j] = t_bin * t_res  
        total_loss[i, j] = s_res[i, j] * t_res_loss[i, j]  

bit_rates = [2**6 * 2, 2**8 * 2, 2**12 * 2]
count_6bit = comb_px * bit_rates[0]
count_8bit = comb_px * bit_rates[1]
count_12bit = comb_px * bit_rates[2]

fig, axs = plt.subplots(2, 2, figsize=(3, 4))

cmap = 'copper'

def add_labels(ax, data):
    for (i, j), val in np.ndenumerate(data):
        ax.text(j, i, f'{val:.1e}', ha='center', va='center', color='white', fontsize=8)

im1 = axs[0, 0].imshow(count_6bit, cmap=cmap, origin='lower', aspect='auto')
axs[0, 0].set_xticks(np.arange(len(s_bins)))
axs[0, 0].set_xticklabels([f'{2*s+1}x{2*s+1}' for s in s_bins])
axs[0, 0].set_yticks(np.arange(len(t_bins)))
axs[0, 0].set_yticklabels([f'{t}' for t in t_bins])
axs[0, 0].set_xlabel('Spatial Binning (Kernel Size)')
axs[0, 0].set_ylabel('Temporal Binning (NFrames)')
axs[0, 0].set_title('Counts (6-bit)')
add_labels(axs[0, 0], count_6bit)

im2 = axs[0, 1].imshow(count_8bit, cmap=cmap, origin='lower', aspect='auto')
axs[0, 1].set_xticks(np.arange(len(s_bins)))
axs[0, 1].set_xticklabels([f'{2*s+1}x{2*s+1}' for s in s_bins])
axs[0, 1].set_yticks(np.arange(len(t_bins)))
axs[0, 1].set_yticklabels([f'{t}' for t in t_bins])
axs[0, 1].set_xlabel('Spatial Binning (Kernel Size)')
axs[0, 1].set_ylabel('Temporal Binning (NFrames)')
axs[0, 1].set_title('Counts (8-bit)')
add_labels(axs[0, 1], count_8bit)

im3 = axs[1, 0].imshow(count_12bit, cmap=cmap, origin='lower', aspect='auto')
axs[1, 0].set_xticks(np.arange(len(s_bins)))
axs[1, 0].set_xticklabels([f'{2*s+1}x{2*s+1}' for s in s_bins])
axs[1, 0].set_yticks(np.arange(len(t_bins)))
axs[1, 0].set_yticklabels([f'{t}' for t in t_bins])
axs[1, 0].set_xlabel('Spatial Binning (Kernel Size)')
axs[1, 0].set_ylabel('Temporal Binning (NFrames)')
axs[1, 0].set_title('Counts (12-bit)')
add_labels(axs[1, 0], count_12bit)

im4 = axs[1, 1].imshow(total_loss, cmap='Reds', origin='lower', aspect='auto')
axs[1, 1].set_xticks(np.arange(len(s_bins)))
axs[1, 1].set_xticklabels([f'{2*s+1}x{2*s+1}' for s in s_bins])
axs[1, 1].set_yticks(np.arange(len(t_bins)))
axs[1, 1].set_yticklabels([f'{t}' for t in t_bins])
axs[1, 1].set_xlabel('Spatial Binning (Kernel Size)')
axs[1, 1].set_ylabel('Temporal Binning (NFrames)')
axs[1, 1].set_title('Total Resolution Loss (nm*ms)')
fig.colorbar(im4, ax=axs[1, 1], label='Total Loss (nm*ms)')

plt.show()
