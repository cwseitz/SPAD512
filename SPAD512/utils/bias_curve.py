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
total_img = 2 ** bits - 1
numsteps = int(1e3 / (freq * step))
times = np.arange(numsteps) * step + offset

def comp_prob(t, lts, wgt, zeta):
    p = np.zeros(len(t))
    for i, lt in enumerate(lts):
        lam = 1 / lt
        p += wgt[i] * zeta * (np.exp(-lam * t) - np.exp(-lam * (t + width)))
    return p

# Simulating mean biases across integration times
integ_times = np.linspace(100, 15000, 200)
mean_biases = np.zeros((len(integ_times)))

for i, integ in enumerate(integ_times):
    numgates = freq * integ
    bin_gates = numgates / total_img
    prob = comp_prob(times, lts, wgt, zeta)
    P_bin = 1 - (1 - prob) ** bin_gates
    counts = np.random.binomial(total_img, P_bin)
    est_prob_bin = counts / total_img
    est_prob_corr = 1 - (1 - est_prob_bin) ** (1 / bin_gates)
    Var_P_bin = (P_bin * (1 - P_bin)) / total_img
    with np.errstate(divide='ignore', invalid='ignore'):
        gpp = (1 / bin_gates) * ((1 / bin_gates) - 1) * (1 - P_bin) ** ((1 / bin_gates) - 2)
    Bias = (gpp / 2) * Var_P_bin
    with np.errstate(divide='ignore', invalid='ignore'):
        Bias_over_prob = Bias / prob
    mean_biases[i] += np.mean(Bias_over_prob)

# Plotting the result with a line at 10 ms
plt.figure()
plt.plot((integ_times / 1000), mean_biases, label='Mean Bias')
plt.axvline(x=10, color='r', linestyle='--', label='Poor results (10 ms)')
plt.xlabel('Integration times (ms)')
plt.ylabel('Mean relative bias')
plt.title('Integration Times versus Relative Bias')
plt.grid(True)
plt.legend()
plt.show()
