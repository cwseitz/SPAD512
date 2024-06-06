import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, poisson
from skimage.io import imread

class PoissonBinomial:
    def __init__(self,p_prior_mean=0.04,p_prior_std=0.05,N_prior_mean=3.0):
        self.p_prior_mean = p_prior_mean
        self.p_prior_std = p_prior_std
        self.N_prior_mean = N_prior_mean
        
    def simulate_data(self,N,p,B,nframes=1000):
        true_counts = np.random.binomial(N, p, size=nframes)
        bg_counts = np.random.poisson(B, size=nframes)
        counts = true_counts + bg_counts
        return counts
    def mc_poisson_binomial(self,N,p,B,nsamples=50000,max_counts=10,plot=False):
        true_counts = np.random.binomial(N, p, size=nsamples)
        bg_counts = np.random.poisson(B, size=nsamples)
        counts = true_counts + bg_counts
        bins = np.arange(0,max_counts+1,1)
        vals, _ = np.histogram(counts, bins=bins, density=True)
        if plot:
            plt.bar(bins[:-1],vals); plt.show()
        return vals

    def likelihood(self, observed_counts, B=0.02, max_counts=10, max_N=10, num_p=100):
        log_likelihoods = np.zeros((max_N, num_p))
        pvec = np.linspace(0,1,num_p)
        for n in range(1,max_N+1):
            for i, p in enumerate(pvec):
                pmf = self.mc_poisson_binomial(n,p,B,max_counts=max_counts)
                log_likelihood = np.sum(np.log(pmf[observed_counts]))
                log_likelihoods[n-1, i] = log_likelihood

        max_log_likelihood = np.max(log_likelihoods)
        log_likelihoods -= max_log_likelihood
        likelihoods = np.exp(log_likelihoods)
        likelihoods = likelihoods/np.sum(likelihoods)

        return likelihoods

    def prior_p(self,p):
        return norm.pdf(p,self.p_prior_mean,self.p_prior_std)

    def prior_N(self,n):
        return poisson.pmf(n,self.N_prior_mean)

    def posterior(self, counts, B=0.02, max_counts=10, max_N=10, num_p=20):
        like = self.likelihood(counts,max_counts=max_counts,
                               B=B,max_N=max_N, num_p=num_p)
        prior = np.ones(max_N) / max_N
        posterior = np.zeros((max_N, num_p))
        pvec = np.linspace(0,1,num_p)
        
        for n in range(1, max_N+1):
            for i, p in enumerate(pvec):
                prior_prob_p = self.prior_p(p)
                prior_prob_N = self.prior_N(n)
                posterior[n-1, i] =\
                np.sum(like[n-1, :] * prior_prob_p * prior_prob_N)

        posterior /= np.sum(posterior)
        return posterior


B = 0.02
max_N = 20
num_p = 20
p_prior_mean = 0.08
p_prior_std = 0.05
N_prior_mean = 1.0

model = PoissonBinomial(p_prior_mean=p_prior_mean,p_prior_std=p_prior_std,
                        N_prior_mean=N_prior_mean)
path = '/research2/shared/cwseitz/Data/SPAD/240604/data/intensity_images/'
file = '240604_SPAD-QD-500kHz-50kHz-1us-350uW-1-trimmed-snip2.tif'
stack = imread(path+file)
counts = np.sum(stack,axis=(1,2))

plt.plot(counts); plt.show()
post = model.posterior(counts, max_N=max_N, max_counts=30, B=B, num_p=num_p)

fig,ax=plt.subplots()
im = plt.imshow(post, origin='lower', extent=[0,1,1,max_N], aspect='auto')
ax.set_xlabel('Probability of emission (p)')
ax.set_ylabel('Number of fluorescent molecules (N)')
plt.colorbar(im, ax=ax, label='Likelihood')

fig,ax=plt.subplots(1,2,figsize=(6,3))
pvec = np.linspace(0,1,num_p)
nvec = np.arange(1,max_N+1,1)
ax[0].bar(pvec,np.sum(post,axis=0),width=0.035)
ax[0].set_xlabel(r'$p$')
ax[0].set_ylabel('Marginal probability')
ax[1].bar(nvec,np.sum(post,axis=1))
ax[1].set_xlabel(r'$N$')
ax[1].set_ylabel('Marginal probability')

plt.tight_layout()
plt.show()


