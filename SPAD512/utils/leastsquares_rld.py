import numpy as np
import aesara.tensor as at
import pymc3 as pm
import matplotlib.pyplot as plt

data = [102, 47, 23, 12]
# data = [47, 23, 12, 6]
# data = [23, 12, 6, 4]
dt = 10
g = 10

def bi_rld_pymc3(D0, D1, D2, D3, dt, g):
    R = D1 * D1 - D2 * D0
    P = D3 * D0 - D2 * D1
    Q = D2 * D2 - D3 * D1
    disc = P ** 2 - 4 * R * Q
    y = (-P + at.sqrt(disc)) / (2 * R)
    x = (-P - at.sqrt(disc)) / (2 * R)
    S = dt * ((x ** 2) * D0 - (2 * x * D1) + D2)
    T = (1 - ((x * D1 - D2) / (x * D0 - D1))) ** (g / dt)
    tau1 = -dt / at.log(y)
    tau2 = -dt / at.log(x)
    A1 = (-(x * D0 - D1) ** 2) * at.log(y) / (S * T)
    A2 = (-R * at.log(x)) / (S * ((x ** (g / dt)) - 1))
    return A1, tau1, A2, tau2

def bayesian_inference(data, dt, g):
    D0, D1, D2, D3 = data

    with pm.Model() as model:
        # Priors for the observed data points
        D0_obs = pm.Normal('D0', mu=D0, sigma=0.1)
        D1_obs = pm.Normal('D1', mu=D1, sigma=0.1)
        D2_obs = pm.Normal('D2', mu=D2, sigma=0.1)
        D3_obs = pm.Normal('D3', mu=D3, sigma=0.1)
        
        # Calculate A1, tau1, A2, tau2 using the provided formula
        A1, tau1, A2, tau2 = bi_rld_pymc3(D0_obs, D1_obs, D2_obs, D3_obs, dt, g)

        # Define likelihood
        sigma = pm.HalfNormal('sigma', sigma=1)
        observed = pm.Normal('observed', mu=[A1, tau1, A2, tau2], sigma=sigma, observed=[A1, tau1, A2, tau2])

        # Inference
        trace = pm.sample(2000, return_inferencedata=True)

    return trace

# Perform Bayesian inference
trace = bayesian_inference(data, dt, g)

# Plot the results
pm.plot_trace(trace)
plt.show()
