import pymc as pm
import numpy as np
import arviz as az
import scipy.special as sp
import matplotlib.pyplot as plt
import aesara
import aesara.tensor as at
import IPython

'''ground truth values (not known)'''
lam1 = 1/20
lam2 = 1/5
A = 0.1
B = 0.5
chi = 1e-4

'''model constants (known in experiment)'''
K = 10000 # number of laser pulses per step
numsteps = 20 # number of steps
step = 5 # ns
offset = 0.018 # ns
width = 5 # ns
tau_irf = 1.5
sigma_irf = 0.5



'''define helper functions for likelihood'''
def h(t, tau_irf, sigma_irf, B, lam1, lam2):
    term1 = B * lam1 * np.exp(lam1 * (tau_irf - t) + 0.5 * lam1**2 * sigma_irf**2) \
            * sp.erfc((tau_irf - t - lam1 * sigma_irf**2) / (sigma_irf * np.sqrt(2)))
    term2 = (1 - B) * lam2 * np.exp(lam2 * (tau_irf - t) + 0.5 * lam2**2 * sigma_irf**2) \
            * sp.erfc((tau_irf - t - lam2 * sigma_irf**2) / (sigma_irf * np.sqrt(2)))
    return term1 + term2

def P_i(start, end, A, B, lam1, lam2, tau_irf, sigma_irf):
    t_vals = np.linspace(start, end, 100)  # for trapezioidal sum, quad integration probably not needed
    h_vals = h(t_vals, tau_irf, sigma_irf, B, lam1, lam2)
    return A * np.trapz(h_vals, t_vals)



'''generate perfect data'''
def gen(K, numsteps, step, offset, width, tau_irf, sigma_irf, A, B, lam1, lam2, chi):
    P_chi = 1 - np.exp(-chi)
    data = []

    for i in range(numsteps):
        start = offset + i*step
        end = start + width

        t_vals = np.linspace(start, end, 100)
        h_vals = h(t_vals, tau_irf, sigma_irf, B, lam1, lam2)
        P_i = A * np.trapz(h_vals, t_vals)

        P_tot = P_i + P_chi

        data.append(np.random.binomial(K, P_tot))

    return data
data = gen(K, numsteps, step, offset, width, tau_irf, sigma_irf, A, B, lam1, lam2, chi)



'''actually relevant code'''
with pm.Model() as model:
    A = pm.HalfNormal('A', sigma=0.2)
    B = pm.Beta('B', alpha=1, beta=1)
    lam1 = pm.Gamma('lam1', alpha=1, beta=1)
    lam2 = pm.Gamma('lam2', alpha=1, beta=1)
    chi = pm.Exponential('chi', lam=1000)

    P_chi = pm.math.exp(-chi)

    # this is the annoying part, need to blackbox the likelihood calculation because its ugly
    P_tot = [P_i(offset + (i-1)*step, offset + (i-1)*step + width, A, B, lam1, lam2, tau_irf, sigma_irf) + P_chi for i in range(numsteps)]
    y_like = [pm.Binomial(f"y_{i}", n=K, p=P_tot[i], observed = data[i]) for i in range(numsteps)]

    # only continuous variables are input so it should default to NUTS
    trace = pm.sample(1000, tune=500, target_accept=0.9, chains=1)


