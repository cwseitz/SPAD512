import numpy as np
import matplotlib.pyplot as plt

def f(N, K, P):
    term1 = -1 + 2**N - K
    term2 = -1 + (1 - P)**(K / (2**N - 1))
    term3 = ((1 - P)**(K / (2**N - 1)))**((2**N - 1 - K) / K)
    return (term1 * term2 * term3) / (2 * K**2)

# Define the ranges for N and K
n = np.linspace(1, 16, 16)
k = np.linspace(1, 200, 400)
N, K = np.meshgrid(n, k)

# Define 5 values of P
P_values = [0.0001, 0.0005, 0.001, 0.005, 0.01]

# Create subplots (1x5 grid)
fig, axes = plt.subplots(1, 5, figsize=(20, 4))  # 1x5 grid, larger figure width
axes = axes.ravel()

for i, ax in enumerate(axes):
    P = P_values[i]
    F = f(N, K, P)
    
    # Calculate relative bias, F/P
    relative_bias = F / P
    
    # Handle any invalid values in relative_bias
    relative_bias = np.nan_to_num(relative_bias, nan=np.nan, posinf=np.nan, neginf=np.nan)
    relative_bias_masked = np.ma.array(relative_bias, mask=np.isnan(relative_bias))
    
    # Plot relative bias
    cp = ax.contourf(N, K, relative_bias_masked, levels=50, cmap='viridis')
    ax.set_title(f'P = {P:.5f}')
    
    fig.colorbar(cp, ax=ax)
    ax.set_xlabel('N')
    ax.set_ylabel('K')

plt.tight_layout()
plt.show()
