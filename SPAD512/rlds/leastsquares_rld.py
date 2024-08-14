import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import matplotlib.pyplot as plt

tfd = tfp.distributions

data = [102, 47, 23, 12]
dt = 10
g = 10

def bi_rld(D0, D1, D2, D3, dt, g):
    R = D1 * D1 - D2 * D0
    P = D3 * D0 - D2 * D1
    Q = D2 * D2 - D3 * D1
    disc = P ** 2 - 4 * R * Q
    y = (-P + tf.sqrt(disc)) / (2 * R)
    x = (-P - tf.sqrt(disc)) / (2 * R)
    S = dt * ((x ** 2) * D0 - (2 * x * D1) + D2)
    T = (1 - ((x * D1 - D2) / (x * D0 - D1))) ** (g / dt)
    tau1 = -dt / tf.math.log(y)
    tau2 = -dt / tf.math.log(x)
    A1 = (-(x * D0 - D1) ** 2) * tf.math.log(y) / (S * T)
    A2 = (-R * tf.math.log(x)) / (S * ((x ** (g / dt)) - 1))
    return A1, tau1, A2, tau2

def bayes(data, dt, g):
    D0, D1, D2, D3 = data

    D0_prior = tfd.Normal(loc=D0, scale=0.1)
    D1_prior = tfd.Normal(loc=D1, scale=0.1)
    D2_prior = tfd.Normal(loc=D2, scale=0.1)
    D3_prior = tfd.Normal(loc=D3, scale=0.1)

    D0_sample = D0_prior.sample()
    D1_sample = D1_prior.sample()
    D2_sample = D2_prior.sample()
    D3_sample = D3_prior.sample()

    A1, tau1, A2, tau2 = bi_rld(D0_sample, D1_sample, D2_sample, D3_sample, dt, g)

    sigma = tfd.HalfNormal(scale=1.0)
    observed = tfd.Normal(loc=[A1, tau1, A2, tau2], scale=sigma)

    def target_log_prob_fn(D0, D1, D2, D3):
        return observed.log_prob([A1, tau1, A2, tau2])

    @tf.function
    def sample():
        return tfp.mcmc.sample_chain(
            num_results=2000,
            current_state=[D0_sample, D1_sample, D2_sample, D3_sample],
            kernel=tfp.mcmc.HamiltonianMonteCarlo(
                target_log_prob_fn=target_log_prob_fn,
                step_size=0.1,
                num_leapfrog_steps=3),
            trace_fn=lambda _, pkr: pkr.inner_results.accepted_results.target_log_prob)

    trace = sample()
    return trace

trace = bayes(data, dt, g)

for i, var_trace in enumerate(trace):
    plt.figure()
    plt.plot(var_trace)
    plt.title(f'Trace of Variable {i+1}')
plt.show()
