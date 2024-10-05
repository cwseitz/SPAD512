import numpy as np
import matplotlib.pyplot as plt

def mono(x, amp, lam):
    return amp * np.exp(-x * lam)

def bi(x, A, lam1, B, lam2):
    return A * (B * np.exp(-x * lam1) + (1 - B) * np.exp(-x * lam2))

file_corr = "C:\\Users\\ishaa\\Documents\\FLIM\\ManFigs\\bit_demo\\corr_fit_results.npz"
file_uncorr = "C:\\Users\\ishaa\\Documents\\FLIM\\ManFigs\\bit_demo\\uncorr_fit_results.npz"

data_corr = np.load(file_corr)
data_uncorr = np.load(file_uncorr)

trace_corr = data_corr['full_trace']
params_corr = data_corr['full_params']
tau1_corr = data_corr['tau1'].flatten()
tau2_corr = data_corr['tau2'].flatten()
times = data_corr['times']

trace_uncorr = data_uncorr['full_trace']
params_uncorr = data_uncorr['full_params']
tau1_uncorr = data_uncorr['tau1'].flatten()
tau2_uncorr = data_uncorr['tau2'].flatten()

tau1_corr = tau1_corr[tau1_corr > 0]
tau2_corr = tau2_corr[tau2_corr > 0]
tau1_uncorr = tau1_uncorr[tau1_uncorr > 0]
tau2_uncorr = tau2_uncorr[tau2_uncorr > 0]

fig, ax = plt.subplots(2, 2, figsize=(10, 8))

bins = 10  
ax[0, 0].hist(tau1_uncorr, bins=bins, alpha=0.5, label='τ₁', color='darkred')
ax[0, 0].hist(tau2_uncorr, bins=bins, alpha=0.5, label='τ₂', color='lightcoral')

ax[0, 0].text(5, 80, "Many small τ₂ values\nnot visible", color="black", fontsize=10, ha="center")

ax[0, 0].set_ylim(0, 100)
ax[0, 0].set_xlabel('Lifetime (ns)')
ax[0, 0].set_ylabel('Counts')
ax[0, 0].set_title('Uncorrected Histogram (Clipped)')
ax[0, 0].legend()

ax[0, 1].scatter(times, trace_uncorr, s=5, label='Uncorrected trace', color='blue')
fit_uncorr = bi(times, *params_uncorr)
tau1_uncorr_fit = 1 / params_uncorr[1]
tau2_uncorr_fit = 1 / params_uncorr[3]
ax[0, 1].plot(times, fit_uncorr, label=f'Fit: τ₁ = {tau1_uncorr_fit:.2f} ns, τ₂ = {1e7*tau2_uncorr_fit:.2f}*1e-7 ns', color='black')
ax[0, 1].set_ylim(0, 1.5 * np.max(trace_uncorr))
ax[0, 1].set_xlabel('Time (ns)')
ax[0, 1].set_ylabel('Counts')
ax[0, 1].set_title('Uncorrected Trace')
ax[0, 1].legend()

ax[1, 0].hist(tau1_corr, bins=bins, alpha=0.5, label='τ₁', color='darkgreen', density=True)
ax[1, 0].hist(tau2_corr, bins=bins, alpha=0.5, label='τ₂', color='lightgreen', density=True)
ax[1, 0].set_xlabel('Lifetime (ns)')
ax[1, 0].set_ylabel('Density')
ax[1, 0].set_title('Corrected Histogram (Unclipped)')
ax[1, 0].legend()

ax[1, 1].scatter(times, trace_corr, s=5, label='Corrected trace', color='orange')
fit_corr = bi(times, *params_corr)
tau1_corr_fit = 1 / params_corr[1]
tau2_corr_fit = 1 / params_corr[3]
ax[1, 1].plot(times, fit_corr, label=f'Fit: τ₁ = {tau1_corr_fit:.2f} ns, τ₂ = {tau2_corr_fit:.2f} ns', color='black')
ax[1, 1].set_ylim(0, 1.5 * np.max(trace_corr))
ax[1, 1].set_xlabel('Time (ns)')
ax[1, 1].set_ylabel('Counts')
ax[1, 1].set_title('Corrected Trace')
ax[1, 1].legend()

plt.tight_layout()
plt.savefig('C:\\Users\\ishaa\\Documents\\FLIM\\ManFigs\\bit_demo\\bit_demo.png')
plt.show()
