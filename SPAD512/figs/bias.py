import numpy as np
import matplotlib.pyplot as plt

def f(N, K, P):
    t1 = -1 + 2**N - K
    t2 = -1 + (1 - P)**(K / (2**N - 1))
    t3 = ((1 - P)**(K / (2**N - 1)))**((2**N - 1 - K) / K)
    return (t1 * t2 * t3) / (2 * K**2)

n = np.linspace(1, 16, 16)
k = np.linspace(1, 200, 400)
N, K = np.meshgrid(n, k)
P_vals = [0.0001, 0.0005, 0.001, 0.005, 0.01]

fig, axs = plt.subplots(1, 5, figsize=(20, 4))
axs = axs.ravel()

for i, ax in enumerate(axs):
    P = P_vals[i]
    F = f(N, K, P)
    rel_bias = F / P
    rel_bias = np.nan_to_num(rel_bias, nan=np.nan, posinf=np.nan, neginf=np.nan)
    rel_bias_masked = np.ma.array(rel_bias, mask=np.isnan(rel_bias))
    cp = ax.contourf(N, K, rel_bias_masked, levels=50, cmap='viridis')
    ax.set_title(f'P = {P:.5f}')
    fig.colorbar(cp, ax=ax)
    ax.set_xlabel('N')
    ax.set_ylabel('K')

plt.tight_layout()
plt.show()
