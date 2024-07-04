import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import binom

def prob_bi_greater_than_or_equal_two(n, zeta):
    """Probability that a draw is greater than or equal to two in any given trial."""
    return 1 - binom.pmf(0, n, zeta) - binom.pmf(1, n, zeta)

def prob_bim_greater_than_or_equal_two(n, zeta):
    """Probability that both elements in the sequence are one or greater."""
    return (1 - binom.pmf(0, n, zeta))**2

def simulate_for_n(n, zeta, nframes, numm, lamb=0.0075):
    B = lamb*nframes*zeta
    prob_zero_lag = prob_bi_greater_than_or_equal_two(n, zeta)
    prob_nonzero_lag = prob_bim_greater_than_or_equal_two(n, zeta)
    num_zero_lag = binom.rvs(nframes, prob_zero_lag)
    nums_nonzero_lag = [binom.rvs(nframes, prob_nonzero_lag) for _ in range(numm)]
    avg_num_nonzero_lag = np.mean(np.array(nums_nonzero_lag))
    return (num_zero_lag-B) / (avg_num_nonzero_lag-B)


nframes = 500000
ns = np.arange(1,10,1)
numm = 100
fig,ax=plt.subplots(figsize=(3,3))

ratios = []
zeta = 0.015
for n in ns:
    ratio = simulate_for_n(n, zeta, nframes, numm)
    ratios.append(ratio)
ratios = np.array(ratios)
ax.scatter(ns, ratios, color='red', marker='x', label=r'$\zeta = 0.01$')

ratios = []
zeta = 0.02
for n in ns:
    ratio = simulate_for_n(n, zeta, nframes, numm)
    ratios.append(ratio)
ratios = np.array(ratios)
ax.scatter(ns, ratios, color='blue', marker='x', label=r'$\zeta = 0.02$')

ratios = []
zeta = 0.05
for n in ns:
    ratio = simulate_for_n(n, zeta, nframes, numm)
    ratios.append(ratio)
ratios = np.array(ratios)
ax.scatter(ns, ratios, color='lime', marker='x', label=r'$\zeta = 0.05$')


ax.set_xlabel('Number of Flurophores (N)')
ax.set_ylabel(r'$g^{(2)}(0)$')
ax.legend()
plt.tight_layout()
plt.show()
