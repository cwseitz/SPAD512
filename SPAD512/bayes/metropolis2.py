import numpy as np
import scipy.special as sp
import scipy.stats as stats
import scipy.integrate as integrate
import matplotlib.pyplot as plt

# acquisition constants
acq_params = {
    'freq': 10,  # laser frequency in MHz
    'numsteps': 10,  # number of gate steps
    'integ': 10000,  # integration time in us
    'width': 5000,  # gate width in ps
    'step': 5000,  # gate step size in ns
    'offset': 18,  # initial gate offset in ps
    'bits': 8,  # bit-rate
    'irf_mean': 0.1,  # tau_IRF in ns
    'irf_width': 1.4  # sigma in ns
}

# ground truth values
theta_true = {
    'lam1': 1 / 20,  # 20 ns
    'lam2': 1 / 5,  # 5 ns
    'A': 0.5,
    'B': 0.5,
    'chi': 0.01  # background count parameter
}

# prior parameters
priors = {
    'alpha_lam1': 1,
    'beta_lam1': 10,
    'alpha_lam2': 1,
    'beta_lam2': 10,
    'alpha_A': 1,
    'beta_A': 1,
    'alpha_B': 1,
    'beta_B': 1,
    'alpha_chi': 0.1,
    'beta_chi': 0.1,
}

# proposal standard deviations
proposals = {
    's_lam1': 0.01,  
    's_lam2': 0.01,
    's_A': 0.05,
    's_B': 0.05,
    's_chi': 0.05,
}

iterations = 5000
burnin = 1000
thin = 1

def h(t, theta, acq_params):
    lam1, lam2, B = theta['lam1'], theta['lam2'], theta['B']
    tau_IRF, sigma = acq_params['irf_mean'], acq_params['irf_width']

    phi1 = lam1 * (tau_IRF - t) + 0.5 * (lam1 * sigma) ** 2
    z1 = (tau_IRF - t - (lam1 * sigma ** 2)) / (sigma * np.sqrt(2))
    h1 = lam1 * np.exp(phi1) * sp.erfc(z1)

    phi2 = lam2 * (tau_IRF - t) + 0.5 * (lam2 * sigma) ** 2
    z2 = (tau_IRF - t - (lam2 * sigma ** 2)) / (sigma * np.sqrt(2))
    h2 = lam2 * np.exp(phi2) * sp.erfc(z2)

    return B * h1 + (1 - B) * h2

def P_i(theta, acq_params):
    A, K, del_t, t_w, t0 = theta['A'], acq_params['numsteps'], acq_params['step'], acq_params['width'], acq_params['offset']
    starts, ends = t0 + np.arange(K) * del_t, t0 + np.arange(K) * del_t + t_w

    P_i_list = [integrate.quad(h, start, end, args=(theta, acq_params), limit=100)[0] for start, end in zip(starts, ends)]
    return A * np.array(P_i_list)

def P_tot_i(P_i_array, chi):
    P_chi = 1 - np.exp(-chi)
    return P_i_array + P_chi

def log_like(y_i, K, P_tot_i_array):
    log_L = y_i * np.log(P_tot_i_array) + (K - y_i) * np.log(1 - P_tot_i_array)
    return np.sum(log_L)

def log_prior(param, alpha, beta):
    return (alpha - 1) * np.log(param) - beta * param

def acceptance_prob(y_i, K, theta, theta_prime, param, acq_params, priors):
    P_i_current = P_i(theta, acq_params)
    P_tot_current = P_tot_i(P_i_current, theta['chi'])
    P_i_prime = P_i(theta_prime, acq_params)
    P_tot_prime = P_tot_i(P_i_prime, theta['chi'])

    log_like_current = log_like(y_i, K, P_tot_current)
    log_like_prime = log_like(y_i, K, P_tot_prime)
    log_prior_current = log_prior(param, priors[f'alpha_{param}'], priors[f'beta_{param}'])
    log_prior_prime = log_prior(theta_prime[param], priors[f'alpha_{param}'], priors[f'beta_{param}'])

    return (log_like_prime + log_prior_prime) - (log_like_current + log_prior_current)





def sample_lam1(theta, y_i, K, priors, acq_params, s_lam1):
    lam1_current = theta['lam1']
    lam1_prime = lam1_current * np.exp(np.random.normal(0, s_lam1))
    theta_prime = theta.copy()
    theta_prime['lam1'] = lam1_prime

    log_alpha = acceptance_prob(y_i, K, theta, theta_prime, 'lam1', acq_params, priors)
    if np.log(np.random.uniform(0, 1)) < log_alpha:
        theta['lam1'] = lam1_prime

def sample_lam2(theta, y_i, K, priors, acq_params, s_lam2):
    lam2_current = theta['lam2']
    lam2_prime = lam2_current * np.exp(np.random.normal(0, s_lam2))
    theta_prime = theta.copy()
    theta_prime['lam2'] = lam2_prime

    log_alpha = acceptance_prob(y_i, K, theta, theta_prime, 'lam2', acq_params, priors)
    if np.log(np.random.uniform(0, 1)) < log_alpha:
        theta['lam2'] = lam2_prime

def sample_A(theta, y_i, K, priors, acq_params, s_A):
    A_current = theta['A']
    logit_A_current = np.log(A_current / (1 - A_current))
    logit_A_prime = logit_A_current + np.random.normal(0, s_A)
    A_prime = 1 / (1 + np.exp(-logit_A_prime))
    theta_prime = theta.copy()
    theta_prime['A'] = A_prime

    log_alpha = acceptance_prob(y_i, K, theta, theta_prime, 'A', acq_params, priors)
    if np.log(np.random.uniform(0, 1)) < log_alpha:
        theta['A'] = A_prime

def sample_B(theta, y_i, K, priors, acq_params, s_B):
    B_current = theta['B']
    logit_B_current = np.log(B_current / (1 - B_current))
    logit_B_prime = logit_B_current + np.random.normal(0, s_B)
    B_prime = 1 / (1 + np.exp(-logit_B_prime))
    theta_prime = theta.copy()
    theta_prime['B'] = B_prime

    log_alpha = acceptance_prob(y_i, K, theta, theta_prime, 'B', acq_params, priors)
    if np.log(np.random.uniform(0, 1)) < log_alpha:
        theta['B'] = B_prime

def sample_chi(theta, y_i, K, priors, acq_params, s_chi):
    chi_current = theta['chi']
    chi_prime = chi_current * np.exp(np.random.normal(0, s_chi))
    theta_prime = theta.copy()
    theta_prime['chi'] = chi_prime

    log_alpha = acceptance_prob(y_i, K, theta, theta_prime, 'chi', acq_params, priors)
    if np.log(np.random.uniform(0, 1)) < log_alpha:
        theta['chi'] = chi_prime





def main_mcmc(iterations, burnin, thin, theta_current, y_i, K, priors, acq_params, proposals):
    num_samples = (iterations - burnin) // thin
    samples = {param: np.zeros(num_samples) for param in theta_current.keys()}

    for iteration in range(iterations):
        sample_lam1(theta_current, y_i, K, priors, acq_params, proposals['s_lam1'])
        sample_lam2(theta_current, y_i, K, priors, acq_params, proposals['s_lam2'])
        sample_A(theta_current, y_i, K, priors, acq_params, proposals['s_A'])
        sample_B(theta_current, y_i, K, priors, acq_params, proposals['s_B'])
        sample_chi(theta_current, y_i, K, priors, acq_params, proposals['s_chi'])

        if iteration >= burnin and (iteration - burnin) % thin == 0:
            index = (iteration - burnin) // thin
            for param in theta_current:
                samples[param][index] = theta_current[param]
    
    return samples


theta_current = {
    'lam1': np.random.gamma(priors['alpha_lam1'], 1 / priors['beta_lam1']),
    'lam2': np.random.gamma(priors['alpha_lam2'], 1 / priors['beta_lam2']),
    'A': 0.5,
    'B': 0.5,
    'chi': 0.01,
}

# generate perfect data
K = int(acq_params['freq'] * acq_params['integ'])
P_i_array_true = P_i(theta_true, acq_params)
P_tot_array_true = P_tot_i(P_i_array_true, theta_true['chi'])
y_i = np.random.binomial(K, P_tot_array_true)

samples = main_mcmc(iterations, burnin, thin, theta_current, y_i, K, priors, acq_params, proposals)

plt.figure(figsize=(12, 8))
for i, param in enumerate(samples):
    plt.subplot(2, 3, i+1)
    plt.hist(samples[param], bins=30, density=True)
    plt.title(f'Posterior of {param}')

plt.tight_layout()
plt.show()
