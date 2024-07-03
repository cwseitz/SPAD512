import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import binom

def prob_bi_greater_than_or_equal_two(n, zeta):
    """Probability that a draw is greater than or equal to two in any given trial."""
    return 1 - binom.pmf(0, n, zeta) - binom.pmf(1, n, zeta)

def prob_bim_greater_than_or_equal_two(n, zeta):
    """Probability that both elements in the sequence are one or greater."""
    return (1 - binom.pmf(0, n, zeta))**2

def simulate_for_n(n, zeta, nframes, numm, Brate=3e-4):
    B = Brate*nframes #expected number of background counts per 1us frame
    prob_zero_lag = prob_bi_greater_than_or_equal_two(n, zeta)
    prob_nonzero_lag = prob_bim_greater_than_or_equal_two(n, zeta)
    num_zero_lag = binom.rvs(nframes, prob_zero_lag)
    nums_nonzero_lag = [binom.rvs(nframes, prob_nonzero_lag) for _ in range(numm)]
    avg_num_nonzero_lag = np.mean(np.array(nums_nonzero_lag))
    return (num_zero_lag-B) / (avg_num_nonzero_lag-B)

zeta = 0.02  # Emission probability per particle
nframes = 500000
ns = np.arange(1,50,1)
numm = 100
ratios = []

for n in ns:
    ratio = simulate_for_n(n, zeta, nframes, numm)
    ratios.append(ratio)

ratios = np.array(ratios)
plt.plot(ns, ratios, color='red')
plt.xlabel('Number of Particles (N)')
plt.ylabel(r'$g^{(2)}(0) = \frac{G^{2}(0)-B}{\langle G^{2}(m)\rangle-B}$')
plt.show()
