import numpy as np
import matplotlib.pyplot as plt

# Your simulation parameters and functions
bits = 8
freq = 10
width_ps = 10000
step_ps = 500
offset_ps = 5000
lts = [20, 5]
wgt = [0.5, 0.5]
zeta = 0.05
dark_cps = 25

width = width_ps * 1e-3
step = step_ps * 1e-3
offset = offset_ps * 1e-3
bitmax = 2 ** bits - 1
numsteps = int(1e3 / (freq * step))
times = np.arange(numsteps) * step + offset

def comp_prob(t, lts, wgt, zeta):
    p = np.zeros(len(t))
    for i, lt in enumerate(lts):
        lam = 1 / lt
        p += wgt[i] * zeta * (np.exp(-lam * t) - np.exp(-lam * (t + width)))
    return p

# Simulating mean biases and skews across integration times
integ_times = np.linspace(25, 15000, 200)
mean_biases = np.zeros((len(integ_times)))
mean_diff = np.zeros((len(integ_times)))
fake_bias = np.zeros((len(integ_times)))

for i, integ in enumerate(integ_times):
    numgates = freq * integ
    bin_gates = numgates / bitmax
    prob = comp_prob(times, lts, wgt, zeta)
    
    # Standard simulation without bit-depth limits
    P_bin = 1 - (1 - prob) ** bin_gates
    counts = np.random.binomial(bitmax, P_bin)
    est_prob_bin = counts / bitmax
    est_prob_corr = 1 - (1 - est_prob_bin) ** (1 / bin_gates)
    Var_P_bin = (P_bin * (1 - P_bin)) / bitmax
    with np.errstate(divide='ignore', invalid='ignore'):
        gpp = (1 / bin_gates) * ((1 / bin_gates) - 1) * (1 - P_bin) ** ((1 / bin_gates) - 2)
    Bias = (gpp / 2) * Var_P_bin
    with np.errstate(divide='ignore', invalid='ignore'):
        Bias_over_prob = Bias / prob
    mean_biases[i] += np.mean(Bias_over_prob)
    mean_diff[i] += np.mean(prob-P_bin)

    # Simulation with bit-depth limit: capping counts at 2^N - 1
    raw_counts = freq * integ * prob
    counts_limited = np.clip(raw_counts, a_min=0, a_max=bitmax)
    mean_rel_skew = np.mean((counts_limited-raw_counts)/raw_counts)
    fake_bias[i] += np.mean(mean_rel_skew)


import numpy as np
import matplotlib.pyplot as plt

# Assuming `mean_biases`, `mean_diff`, `fake_bias`, `integ_times` are defined from the earlier code.

# Step 1: Find the y-value where x = 10 ms (the blue line)
x_10_ms = 10  # 10 ms
x_10_idx = np.argmin(np.abs(integ_times / 1000 - x_10_ms))  # Index closest to 10 ms
y_at_10ms = mean_biases[x_10_idx]  # Get the y-value at 10 ms for blue curve

# Step 2: Find the corresponding x-value for the green line (mean_diff) at the same y-value
x_poor_uncorrected_idx = np.argmin(np.abs(mean_diff - y_at_10ms))  # Index where green curve intersects the y_at_10ms
x_poor_uncorrected = integ_times[x_poor_uncorrected_idx] / 1000  # Corresponding x-value in ms for green curve

# Create the subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

# Plot 1: Solid lines (blue, green, and black)
ax1.plot((integ_times / 1000), mean_biases, label='Mean Bias', color='blue', linestyle='solid')
ax1.plot((integ_times / 1000), mean_diff, label='Mean Uncorrected Bias', color='green', linestyle='solid')
ax1.plot((integ_times / 1000), fake_bias, label='Mean Bias for Conventional 2^N - 1 Limit', color='black', linestyle='solid')

# Set axis labels and title for Plot 1
ax1.set_xlabel('Integration times (ms)')
ax1.set_ylabel('Mean relative bias')
ax1.set_title('Comparison of Biases')
ax1.grid(True)
ax1.legend()

# Plot 2: Solid and dashed lines (blue, green), plus red dashed line for the intersection
ax2.plot((integ_times / 1000), mean_biases, label='Mean Bias (Corrected)', color='blue', linestyle='solid')
ax2.plot((integ_times / 1000), mean_diff, label='Mean Uncorrected Bias', color='green', linestyle='solid')

# Plot the dashed lines
ax2.axvline(x=10, color='blue', linestyle=(0, (5,10)), label=f'Poor results for Corrected (x=10 ms)')
ax2.axvline(x=x_poor_uncorrected, color='green', linestyle=(0, (5,10)), label=f'Poor results for Uncorrected (x={x_poor_uncorrected:.2f} ms)')
ax2.axhline(y=y_at_10ms, color='red', linestyle=(0, (5,10)), label=f'Intersection y-value = {y_at_10ms:.2e}')

ax2.set_ylim(-0.05, 0)  # Adjusting y-axis around the intersection

# Set axis labels and title for Plot 2
ax2.set_xlabel('Integration times (ms)')
ax2.set_ylabel('Mean relative bias')
ax2.set_title('Determination of Acceptable Bias')
ax2.grid(True)
ax2.legend()

# Show the plot
plt.tight_layout()
plt.show()
