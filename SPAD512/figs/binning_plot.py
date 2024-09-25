import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the .npz files
lma_data = np.load(r"C:\Users\ishaa\Documents\FLIM\ManFigs\lma_fit_results.npz")
nnls_data = np.load(r"C:\Users\ishaa\Documents\FLIM\ManFigs\nnls_fit_results.npz")

# Extract tau1 and tau2 from LMA and filter values less than 3
tau1_lma = lma_data['tau1'].ravel()
tau2_lma = lma_data['tau2'].ravel()
tau1_lma = tau1_lma[tau1_lma >= 3]
tau2_lma = tau2_lma[tau2_lma >= 3]

# Extract tau1 and tau2 from NNLS and filter values less than 3
tau1_nnls = nnls_data['tau1'].ravel()
tau1_nnls = tau1_nnls[tau1_nnls >= 3]
tau2_nnls = nnls_data['tau2'].ravel()
tau2_nnls = tau2_nnls[tau2_nnls >= 3]

# Determine common bin edges for consistent bin width
min_value = min(np.min(tau1_lma), np.min(tau2_lma), np.min(tau1_nnls), np.min(tau2_nnls))
max_value = max(np.max(tau1_lma), np.max(tau2_lma), np.max(tau1_nnls), np.max(tau2_nnls))
bins = np.linspace(min_value, max_value, 200)

# Set seaborn style for a polished look
sns.set_theme(style="white")

# Create the figure and subplots
fig, ax = plt.subplots(2, 1, figsize=(10, 8), sharex=True, sharey=True)

# Plot the stacked histograms for LMA
ax[0].hist([tau1_lma, tau2_lma], bins=bins, color=['red', 'blue'], alpha=0.6, label=['Longer', 'Shorter'], stacked=True)
ax[0].set_title('Fitting without non-negative step', fontweight='bold')
ax[0].set_xlabel('Lifetimes (ns)')
ax[0].set_ylabel('Frequency')
ax[0].legend()

# Plot the stacked histograms for NNLS
ax[1].hist([tau1_nnls, tau2_nnls], bins=bins, color=['red', 'blue'], alpha=0.6, label=['Longer', 'Shorter'], stacked=True)
ax[1].set_title('Fitting with LMA + NNLS', fontweight='bold')
ax[1].set_xlabel('Lifetimes (ns)')
ax[1].set_ylabel('Frequency')
ax[1].legend()

# Final adjustments
for a in ax:
    a.spines['top'].set_visible(False)
    a.spines['right'].set_visible(False)

plt.tight_layout()
plt.show()
