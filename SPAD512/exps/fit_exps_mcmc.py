import numpy as np
import emcee


# Bayesian method for getting double exponential info (incomplete, need to adjust params return)
def like(self, params, times, data):
    model = self.decay_double(times, params[0], params[1], params[2], params[3])
    sigma2 = 0.1**2
    return -0.5 * np.sum((data - model)**2 / sigma2 + np.log(sigma2))

def prior(self, params):
    if any(param <= 0 for param in params):
        return -np.inf
    return 0

def prob(self, params, times, data):
    lp = self.prior(params)
    if not np.isfinite(lp):
        return -np.inf
    return lp + self.like(params, times, data)

def mcmc(self, times, data):
    ndim = 4  # can adjust this if triple exponential is desired
    walkers = 32
    guess = [np.max(data), 2.0, np.max(data) / 2, 1.0] 
    pos = guess + (1e-4 * np.random.randn(walkers, ndim))

    sampler = emcee.EnsembleSampler(walkers, ndim, self.prob, args=(times, data))
    sampler.run_mcmc(pos, 5000, progress=True)

    samples = sampler.get_chain(discard=100, thin=15, flat=True)
    params = np.mean(samples, axis=0)
    return params