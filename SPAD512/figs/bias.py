import numpy as np
import matplotlib.pyplot as plt

def f(N, K, P):
    t1 = -1 + 2**N - K
    t2 = -1 + (1 - P)**(K / (2**N - 1))
    t3 = ((1 - P)**(K / (2**N - 1)))**((2**N - 1 - K) / K)
    return (t1 * t2 * t3) / (2 * K**2)

# n = np.linspace(1, 8, 8)
# k = np.linspace(1, 200, 400)
# N, K = np.meshgrid(n, k)
# P_vals = np.linspace(0.00001, 0.005, 8)
# fig, axs = plt.subplots(2, 4, figsize=(10, 4))
# axs = axs.ravel()

# for i, ax in enumerate(axs):
#     P = P_vals[i]
#     F = f(N, K, P)
#     rel_bias = F / P
#     rel_bias = np.nan_to_num(rel_bias, nan=np.nan, posinf=np.nan, neginf=np.nan)

#     # Clip values to be between 0 and 1
#     rel_bias_clipped = np.clip(rel_bias, 0, 1)

#     rel_bias_masked = np.ma.array(rel_bias_clipped, mask=np.isnan(rel_bias_clipped))
    
#     # Create contour plot with clipped values
#     cp = ax.contourf(N, K, rel_bias_masked, levels=50, cmap='viridis')
#     ax.set_title(f'P = {P:.5f}')
#     fig.colorbar(cp, ax=ax)
#     ax.set_xlabel('N')
#     ax.set_ylabel('K')

# plt.tight_layout()
# plt.show()

# Plot for fixed N = 8 and P = 0.001 as a function of K
N_fixed = 8
P_fixed = 0.015
K_values = np.linspace(10000, 200000, 1000)  # Same range as before

# Compute the function for fixed N and P
F_fixed = f(N_fixed, K_values, P_fixed)
rel_bias_fixed = F_fixed + P_fixed
rel_bias_fixed = np.nan_to_num(rel_bias_fixed, nan=np.nan, posinf=np.nan, neginf=np.nan)

# Plot the results
plt.figure(figsize=(6, 4))
plt.plot(K_values, rel_bias_fixed, label=f'N = {N_fixed}, P = {P_fixed}')
plt.plot(0.015)
plt.xlabel('K')
plt.ylabel('Relative Bias (Clipped)')
plt.title(f'Relative Bias vs K for N = {N_fixed} and P = {P_fixed}')
plt.grid(True)
plt.legend()
plt.show()

print(f(8, 200000, 0.015) + .015)
