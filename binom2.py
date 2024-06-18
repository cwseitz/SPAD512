import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.special import factorial

class PoissonBinomial:
    def __init__(self, x, lambd=1, zeta_mean=0.05, zeta_std=0.01):
        self.x = x
        self.lambd = lambd
        self.zeta_mean = zeta_mean
        self.zeta_std = zeta_std

    def log_poisson_likelihood(self, n, zeta):
        return np.sum(-self.lambd - n * zeta + self.x * np.log(self.lambd + n * zeta) - self.x*np.log(1e-8+self.x) + self.x)

    def gaussian_prior(self, zeta):
        return norm.pdf(zeta, loc=self.zeta_mean, scale=self.zeta_std)

    def monte_carlo_integration(self, num_samples, N):
        integral_sum = 0
        zeta_samples = np.random.uniform(0,1,size=num_samples)
        likes = []; priors = []
        for zeta in zeta_samples:
            log_like = self.log_poisson_likelihood(N,zeta)
            prior = self.gaussian_prior(zeta)
            likes.append(log_like)
            priors.append(prior)
        likes = np.array(likes)
        priors = np.array(priors)
        print(f'Max log-likelihood: {np.max(likes)}')
        likes -= np.max(likes)
        likes = np.exp(likes)
        return np.sum(likes*priors)/num_samples

np.random.seed(42)
N = 4
zeta = 0.05
lambd = 0.02

binomial_data = np.random.binomial(N, zeta, size=1000)
poisson_data = np.random.poisson(lambd, size=1000)
observed_data = binomial_data + poisson_data

poisson_binomial = PoissonBinomial(observed_data,lambd=lambd)

num_samples = 1000
post = [poisson_binomial.monte_carlo_integration(num_samples, n) for n in range(1, 11)]
post = np.array(post)
post = post/np.sum(post)

plt.bar(range(1, 11), post)
plt.xlabel('N')
plt.ylabel('Posterior Probability')
plt.title('Posterior Distribution over N')
plt.show()

