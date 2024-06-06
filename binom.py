import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, poisson

def simulate_data(N, p, B, nframes=1000):
    true_counts = np.random.binomial(N, p, size=nframes)
    bg_counts = np.random.poisson(B, size=nframes)
    counts = true_counts + bg_counts
    return counts
    
def mc_poisson_binomial(N, p, B, nsamples=10000, max_counts=10, plot=False):
    true_counts = np.random.binomial(N, p, size=nsamples)
    bg_counts = np.random.poisson(B, size=nsamples)
    counts = true_counts + bg_counts
    bins = np.arange(0,max_counts+1,1)
    vals, _ = np.histogram(counts, bins=bins, density=True)
    if plot:
        plt.bar(bins[:-1],vals); plt.show()
    return vals

def likelihood_image(observed_counts, max_counts=10, max_N=10, num_p=100):
    log_likelihoods = np.zeros((max_N, num_p))
    pvec = np.linspace(0,1,num_p)
    for n in range(1,max_N+1):
        for i, p in enumerate(pvec):
            pmf = mc_poisson_binomial(n, p, B, max_counts=max_counts)
            log_likelihood = np.sum(np.log(pmf[observed_counts])) # count is also index
            log_likelihoods[n-1, i] = log_likelihood

    max_log_likelihood = np.max(log_likelihoods)
    log_likelihoods -= max_log_likelihood
    likelihoods = np.exp(log_likelihoods)
    likelihoods = likelihoods/np.sum(likelihoods)

    return likelihoods

def prior_p(p, mean=0.2, std=0.01):
    return norm.pdf(p, mean, std)

def prior_N(N, peak=3, decay_rate=0.5):
    return poisson.pmf(N, peak) * np.exp(-decay_rate * (N - peak))

def bayesian_update(observed_counts, max_counts=10, max_N=10, num_p=20):
    likelihoods = likelihood_image(observed_counts, max_counts=max_counts, max_N=max_N, num_p=num_p)
    prior = np.ones(max_N) / max_N  # Uniform prior for N
    posterior = np.zeros((max_N, num_p))
    pvec = np.linspace(0, 1, num_p)
    
    for n in range(1, max_N+1):
        for i, p in enumerate(pvec):
            prior_prob_p = prior_p(p)
            prior_prob_N = prior_N(n)
            posterior[n-1, i] = np.sum(likelihoods[n-1, :] * prior_prob_p * prior_prob_N)

    # Normalize posterior
    posterior /= np.sum(posterior)

    return posterior

N = 2
p = 0.2 # need to measure
B = 0.0 # need to measure
max_N = 20

counts = simulate_data(N, p, B)
plt.plot(counts)
plt.show()

likelihoods = likelihood_image(counts, max_N=max_N, max_counts=30)
posterior = bayesian_update(counts, max_N=max_N, max_counts=30)

fig,ax=plt.subplots()
im = ax.imshow(likelihoods, origin='lower', extent=[0,1,1,max_N], aspect='auto')
ax.set_xlabel('Probability of emission (p)')
ax.set_ylabel('Number of fluorescent molecules (N)')
plt.colorbar(im, ax=ax, label='Likelihood')

fig,ax=plt.subplots()
im = plt.imshow(posterior, origin='lower', extent=[0,1,1,max_N], aspect='auto')
ax.set_xlabel('Probability of emission (p)')
ax.set_ylabel('Number of fluorescent molecules (N)')
plt.colorbar(im, ax=ax, label='Likelihood')

plt.tight_layout()
plt.show()

