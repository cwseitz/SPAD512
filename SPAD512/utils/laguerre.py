import numpy as np
import matplotlib.pyplot as plt
from scipy.special import eval_genlaguerre
from scipy.optimize import least_squares
from math import factorial, sqrt, pi

def gaussian(t, sigma):
    return np.exp(-0.5*(t/sigma)**2) / (sigma * np.sqrt(2*np.pi))

def laguerre_basis(t, n, alpha):
    t_clip = np.clip(t, 0, None)
    Ln = eval_genlaguerre(n, 0, t_clip/alpha)
    return np.exp(-t_clip/(2*alpha)) * Ln

def convolve_numerical(f, g, t_array):
    dt = t_array[1] - t_array[0]
    conv_result = np.zeros_like(t_array)
    for i, t in enumerate(t_array):
        tau_vals = t_array[t_array <= t]
        f_vals = f[t_array <= t]
        g_vals = g[i - np.arange(len(tau_vals))]
        conv_result[i] = np.sum(f_vals*g_vals) * dt
    return conv_result


t_max = 10.0
num_points = 200
t_grid = np.linspace(0, t_max, num_points)
tau_true = 2.0
E_true = (1.0 / tau_true) * np.exp(-t_grid / tau_true)

sigma = 2
R_vals = gaussian(t_grid, sigma)  
G_vals = np.convolve(E_true, R_vals, mode='full')[:num_points] * (t_grid[1] - t_grid[0])

noise_level = 0.01
G_noisy = G_vals + noise_level * np.random.randn(num_points)

N_basis = 5
alpha = 2.0  #  scale param
basis_convolved = np.zeros((num_points, N_basis))

for n in range(N_basis):
    phi_n = laguerre_basis(t_grid, n, alpha)
    psi_n = np.convolve(phi_n, R_vals, mode='full')[:num_points] * (t_grid[1] - t_grid[0])
    basis_convolved[:, n] = psi_n


A = basis_convolved 
g = G_noisy          

a_hat, residuals, rank, svals = np.linalg.lstsq(A, g, rcond=None)

E_reconstructed = np.zeros_like(t_grid)
for n in range(N_basis):
    E_reconstructed += a_hat[n] * laguerre_basis(t_grid, n, alpha)

plt.figure(figsize=(8,5))
plt.plot(t_grid, E_true, 'k--', label='true exponential')
plt.plot(t_grid, E_reconstructed, 'r', label='reconstructed')
plt.plot(t_grid, G_noisy, 'b.', markersize=3, label='convolved')
plt.xlabel('t')
plt.ylabel('amp')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
