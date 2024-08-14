import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve, deconvolve
from scipy.integrate import quad
import random

import warnings
warnings.simplefilter("ignore", category=RuntimeWarning)

def bi_exp(x, A1, t1, A2, t2):
    return A1 * np.exp(-x / t1) + A2 * np.exp(-x / t2)

def integrate(region, A1, tau1, A2, tau2, factor=1000):
    lower, upper = region
    result, _ = quad(bi_exp, lower, upper, args=(A1, tau1, A2, tau2))
    result *= factor
    return result

def bi_rld(data, g, s):
    D0, D1, D2, D3 = data

    R = D1*D1 - D2*D0
    P = D3*D0 - D2*D1
    Q = D2*D2 - D3*D1
    disc = P**2 - 4*R*Q
    y = (-P + np.sqrt(disc))/(2*R)
    x = (-P - np.sqrt(disc))/(2*R)
    S = s * ((x**2)*D0 - (2*x*D1) + D2)
    T = (1-((x*D1 - D2)/(x*D0 - D1))) ** (g/s)

    tau1 = -s/np.log(y)
    tau2 = -s/np.log(x)

    A1 = (-(x*D0 - D1)**2) * np.log(y) / (S * T) 
    A2 = (-R * np.log(x)) / (S * ((x**(g/s)) - 1))

    return (A1, tau1, A2, tau2)

def perturb(data, sigma=0.1):
    data = np.array(data)
    data += data * np.random.normal(0, sigma, size=data.shape)
    return data

def run_single(params, g, s, off, sigma=0.1, factor=1000):
    A1, t1, A2, t2 = params

    regs = [(off, off+g), (off+s, off+g+s), (off+2*s, off+g+2*s), (off+3*s, off+g+3*s)]
    data = []

    for reg in regs:
        counts = integrate(reg, A1, t1, A2, t2, factor=factor)
        data.append(counts)

    data = perturb(data, sigma=sigma)

    n_A1, n_t1, n_A2, n_t2 = bi_rld(data, g, s)
    results = [(t1-n_t1)/t1, (t2-n_t2)/t2]
    norm_t = np.linalg.norm(np.array(results))
    
    # return norm_t
    return(n_t1, n_t2)

params = [1, 5, 2, 20]
g_sims = [5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
s_sims = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
iters = 10
sigmas = np.linspace(0.0001, 0.1, num=1000)
off = 0

finals = []
slopes = np.zeros((len(g_sims), len(s_sims)))
tau1 = np.zeros(iters)
tau2 = np.zeros(iters)
for i, g in enumerate(g_sims):
    for j, s in enumerate(s_sims):
        print(f'Testing gate {g} ns and step {s} ns')
        results = np.zeros(len(sigmas))
        for k, sigma in enumerate(sigmas):
            for l in range(iters):
                tau1[l], tau2[l] = run_single(params, g, s, off, sigma=sigma, factor=10)
            results[k] = np.nanstd(tau1)/(params[1])

        mask = ~np.isnan(results)
        results = results[mask]
        sigmas = sigmas[mask]

        q1 = np.percentile(results, 25)
        q3 = np.percentile(results, 75)
        iqr = q3-q1
        lower = q1 - 1.5*iqr
        upper = q3 + 1.5*iqr

        indices = np.where((results < lower) | (results > upper))
        reg_sigmas = np.delete(sigmas, indices)
        reg_results = np.delete(results, indices)

        slope, intercept = np.polyfit(reg_sigmas, reg_results, 1)

        slopes[i,j] = slope
        print(slope)
        print('')

fig, ax = plt.subplots()
cax = ax.imshow(slopes)
ax.set_title('Delta value over various gate and step sizes')
ax.set_ylabel('Gate length (ns)')
ax.set_xlabel('Step size (ns)')
ax.set_yticks(np.linspace(0, len(g_sims), num=len(g_sims), endpoint=False))
ax.set_yticklabels(g_sims)
ax.set_xticks(np.linspace(0, len(s_sims), num=len(s_sims), endpoint=False))
ax.set_xticklabels(s_sims)
fig.colorbar(cax)
plt.show()
plt.savefig("C://Users//ishaa//Documents//FLIM//240813//sigmas_sim.png")

# fig, ax = plt.subplots()
# cax = ax.imshow(avg)
# ax.set_title('Delta value over various gate and step sizes')
# ax.set_ylabel('Gate length (ns)')
# ax.set_xlabel('Step size (ns)')
# ax.set_yticks(np.linspace(0, len(g_sims), num=len(g_sims), endpoint=False))
# ax.set_yticklabels(g_sims)
# ax.set_xticks(np.linspace(0, len(s_sims), num=len(s_sims), endpoint=False))
# ax.set_xticklabels(s_sims)
# fig.colorbar(cax)
# plt.show()
    

