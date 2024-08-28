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

fig, axs = plt.subplots(1, 2, figsize=(12, 6))

im1 = axs[0].imshow(comb_px, cmap='viridis', origin='lower', aspect='auto')
axs[0].set_xticks(np.arange(len(s_bins)))
axs[0].set_xticklabels([f'{2*s+1}x{2*s+1}' for s in s_bins])
axs[0].set_yticks(np.arange(len(t_bins)))
axs[0].set_yticklabels([f'{t}' for t in t_bins])
axs[0].set_xlabel('Spatial Binning (Kernel Size)')
axs[0].set_ylabel('Temporal Binning (Number of Frames)')
axs[0].set_title('Number of Combined Pixels')
fig.colorbar(im1, ax=axs[0], label='Number of Combined Pixels')

im2 = axs[1].imshow(total_loss, cmap='magma', origin='lower', aspect='auto')
axs[1].set_xticks(np.arange(len(s_bins)))
axs[1].set_xticklabels([f'{2*s+1}x{2*s+1}' for s in s_bins])
axs[1].set_yticks(np.arange(len(t_bins)))
axs[1].set_yticklabels([f'{t}' for t in t_bins])
axs[1].set_xlabel('Spatial Binning (Kernel Size)')
axs[1].set_ylabel('Temporal Binning (Number of Frames)')
axs[1].set_title('Total Resolution Loss (nm*ms)')
fig.colorbar(im2, ax=axs[1], label='Total Loss (nm*ms)')

plt.tight_layout()
plt.show()
