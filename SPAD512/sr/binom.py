import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.special import factorial
from skimage.io import imread

class PoissonBinomial:
    def __init__(self, x, lambd=1, zeta_mean=0.05, zeta_std=0.01):
        self.x = x
        self.lambd = lambd
        self.zeta_mean = zeta_mean
        self.zeta_std = zeta_std

    def monte_log_likelihood(self,N,zeta,nsamples=1000,max_cts=200,plot=False):
        true_counts = np.random.binomial(N,zeta,size=nsamples)
        bg_counts = np.random.poisson(self.lambd,size=nsamples)
        mc_counts = true_counts + bg_counts
        bins = np.arange(0,max_cts,1)
        pmf, _ = np.histogram(self.x, bins=bins, density=True)
        mc_pmf, _ = np.histogram(mc_counts, bins=bins, density=True)
        if plot:
            plt.bar(bins[:-1],pmf,color='red');
            plt.bar(bins[:-1],mc_pmf,color='blue')
            plt.show()
        log_like = np.sum(np.log(1e-8+mc_pmf[self.x]))
        return log_like

    def approx_log_likelihood(self, n, zeta):
        return np.sum(-self.lambd - n * zeta + self.x * np.log(self.lambd + n * zeta) - self.x*np.log(1e-8+self.x) + self.x)

    def gaussian_prior(self, zeta):
        return norm.pdf(zeta, loc=self.zeta_mean, scale=self.zeta_std)

    def integrate(self, num_samples, N, approx=False):
        zeta_samples = np.random.uniform(0,1,size=num_samples)
        likes = []; priors = []
        for zeta in zeta_samples:
            if approx:
                log_like = self.approx_log_likelihood(N,zeta)
            else:
                log_like = self.monte_log_likelihood(N,zeta)
            prior = self.gaussian_prior(zeta)
            likes.append(log_like); priors.append(prior)
        likes = np.array(likes)
        priors = np.array(priors)
        #print(f'Max log-likelihood: {np.max(likes)}')
        likes -= np.max(likes)
        likes = np.exp(likes)
        return np.sum(likes*priors)/num_samples




