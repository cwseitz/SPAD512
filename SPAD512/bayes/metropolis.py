import numpy as np
import scipy.special as sp
import scipy.stats as stats
import scipy.integrate as integrate
import logging
import matplotlib.pyplot as plt

# Set up logging for debugging
logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(message)s')

# Constants (all time units in nanoseconds)
K = 1000        # Number of pulses per gate
N = 50          # Number of gates
t0 = 0.018      # Offset time (in nanoseconds)
Delta_t = 1     # Step size (in nanoseconds)
t_w = 5         # Gate width (in nanoseconds)
sigma = 1.4     # Standard deviation of the Gaussian IRF (in nanoseconds)
tau_IRF = 1.5   # Time delay due to IRF (in nanoseconds)
P_chi = 1e-4    # Background count probability

# Ground truth parameters
lambda1_truth = 1 / 20    # 1/20 ns^-1 (corresponds to a lifetime of 20 ns)
lambda2_truth = 1 / 5     # 1/5 ns^-1 (corresponds to a lifetime of 5 ns)
A_truth = 1.0
B_truth = 0.5

# Create theta dictionary with ground truth parameters
theta_truth = {
    'lambda1': lambda1_truth,
    'lambda2': lambda2_truth,
    'A': A_truth,
    'B': B_truth
}

# Define the h(t) function
def h(t, theta):
    lambda1 = theta['lambda1']
    lambda2 = theta['lambda2']
    B = theta['B']

    # Compute terms for lambda1
    phi1 = lambda1 * (tau_IRF - t) + 0.5 * (lambda1 * sigma) ** 2
    z1 = (tau_IRF - t - (lambda1 * sigma ** 2)) / (sigma * np.sqrt(2))
    h1 = lambda1 * np.exp(phi1) * sp.erfc(z1)

    # Compute terms for lambda2
    phi2 = lambda2 * (tau_IRF - t) + 0.5 * (lambda2 * sigma) ** 2
    z2 = (tau_IRF - t - (lambda2 * sigma ** 2)) / (sigma * np.sqrt(2))
    h2 = lambda2 * np.exp(phi2) * sp.erfc(z2)

    # Combine the two components
    h_t = B * h1 + (1 - B) * h2
    return h_t

# Function to compute P_i for a given gate i
def compute_P_i(i, theta):
    A = theta['A']
    t_start = t0 + (i - 1) * Delta_t
    t_end = t_start + t_w

    # Numerical integration of h(t) over the gate interval
    result, error = integrate.quad(lambda t: h(t, theta), t_start, t_end, limit=100)
    P_i = A * result
    return P_i

# Generate perfect noiseless data
y_obs = np.zeros(N)
x = np.linspace(t0, t0 + (N - 1) * Delta_t + t_w, 1000)
h_values = h(x, theta_truth)

for i in range(1, N + 1):
    P_i = compute_P_i(i, theta_truth)
    P_tot_i = P_i + P_chi
    y_obs[i - 1] = K * P_tot_i  # Perfect noiseless data

# Now proceed with the MCMC algorithm using the observed data y_obs

# Prior parameters
alpha_lambda1 = 2.0
beta_lambda1 = 0.1  # Adjusted for appropriate scale
alpha_lambda2 = 2.0
beta_lambda2 = 0.1  # Adjusted for appropriate scale
alpha_A = 1.0
beta_A = 1.0
alpha_B = 1.0
beta_B = 1.0

# Initialize parameters
theta_init = {
    'lambda1': 0.01,  # Initial guess for lambda1 (1/ns)
    'lambda2': 0.6,   # Initial guess for lambda2 (1/ns)
    'A': 0.9,         # Initial guess for A
    'B': 0.8          # Initial guess for B
}

# Proposal standard deviations
s_lambda1 = 1
s_lambda2 = 1
s_A = 1
s_B = 1   

# Number of iterations
num_iterations = 5000
burn_in = 1000
thin = 1

# Compute the log-likelihood
def log_likelihood(theta):
    log_L = 0.0
    for i in range(1, N + 1):
        P_i = compute_P_i(i, theta)
        P_tot_i = P_i + P_chi

        # Ensure probabilities are within (0,1)
        P_tot_i = min(max(P_tot_i, 1e-10), 1 - 1e-10)

        y_i = y_obs[i - 1]
        # Use Poisson likelihood since counts are from discrete events
        expected_count = K * P_tot_i
        # Small value added to prevent log(0)
        log_L += y_i * np.log(expected_count + 1e-10) - expected_count - sp.gammaln(y_i + 1)

        # Debugging output
        logging.debug(f'Gate {i}: P_i={P_i}, P_tot_i={P_tot_i}, y_i={y_i}, log_L={log_L}')
    return log_L

# Compute the log-prior
def log_prior(theta):
    lambda1 = theta['lambda1']
    lambda2 = theta['lambda2']
    A = theta['A']
    B = theta['B']

    # Log-prior for lambda1
    log_p_lambda1 = (alpha_lambda1 - 1) * np.log(lambda1) - beta_lambda1 * lambda1
    # Log-prior for lambda2
    log_p_lambda2 = (alpha_lambda2 - 1) * np.log(lambda2) - beta_lambda2 * lambda2
    # Log-prior for A
    log_p_A = (alpha_A - 1) * np.log(A) + (beta_A - 1) * np.log(1 - A)
    # Log-prior for B
    log_p_B = (alpha_B - 1) * np.log(B) + (beta_B - 1) * np.log(1 - B)

    log_p = log_p_lambda1 + log_p_lambda2 + log_p_A + log_p_B

    # Debugging output
    logging.debug(f'Log-prior: log_p_lambda1={log_p_lambda1}, log_p_lambda2={log_p_lambda2}, log_p_A={log_p_A}, log_p_B={log_p_B}')
    return log_p

# Reflection function to keep parameters within [0,1] or positive
def reflect(x, lower=0, upper=1):
    while x < lower or x > upper:
        if x < lower:
            x = 2 * lower - x
        elif x > upper:
            x = 2 * upper - x
    return x

# Metropolis-Hastings algorithm
def metropolis_hastings():
    theta_current = theta_init.copy()
    samples = []
    acceptance_count = 0

    for t in range(1, num_iterations + 1):
        theta_proposed = {}

        # Propose new lambda1
        epsilon_lambda1 = np.random.normal(0, s_lambda1)
        theta_proposed['lambda1'] = theta_current['lambda1'] * np.exp(epsilon_lambda1)
        # Ensure positivity
        theta_proposed['lambda1'] = max(theta_proposed['lambda1'], 1e-10)

        # Propose new lambda2
        epsilon_lambda2 = np.random.normal(0, s_lambda2)
        theta_proposed['lambda2'] = theta_current['lambda2'] * np.exp(epsilon_lambda2)
        # Ensure positivity
        theta_proposed['lambda2'] = max(theta_proposed['lambda2'], 1e-10)

        # Propose new A
        eta_A = np.random.normal(0, s_A)
        A_proposed = theta_current['A'] + eta_A
        # Reflect at boundaries
        A_proposed = reflect(A_proposed, 0, 1)
        theta_proposed['A'] = A_proposed

        # Propose new B
        eta_B = np.random.normal(0, s_B)
        B_proposed = theta_current['B'] + eta_B
        # Reflect at boundaries
        B_proposed = reflect(B_proposed, 0, 1)
        theta_proposed['B'] = B_proposed

        # Compute log-likelihood and log-prior for current and proposed
        log_L_current = log_likelihood(theta_current)
        log_L_proposed = log_likelihood(theta_proposed)
        log_p_current = log_prior(theta_current)
        log_p_proposed = log_prior(theta_proposed)

        # Compute proposal ratios for lambda1 and lambda2
        delta_log_q_lambda1 = np.log(theta_proposed['lambda1'] / theta_current['lambda1'])
        delta_log_q_lambda2 = np.log(theta_proposed['lambda2'] / theta_current['lambda2'])
        delta_log_q = delta_log_q_lambda1 + delta_log_q_lambda2

        # Compute acceptance probability
        delta_log_L = log_L_proposed - log_L_current
        delta_log_p = log_p_proposed - log_p_current
        delta_log_alpha = delta_log_L + delta_log_p + delta_log_q

        alpha = min(1, np.exp(delta_log_alpha))

        # Debugging output
        logging.debug(f'Iteration {t}:')
        logging.debug(f'Theta_current: {theta_current}')
        logging.debug(f'Theta_proposed: {theta_proposed}')
        logging.debug(f'log_L_current: {log_L_current}, log_L_proposed: {log_L_proposed}')
        logging.debug(f'log_p_current: {log_p_current}, log_p_proposed: {log_p_proposed}')
        logging.debug(f'delta_log_q: {delta_log_q}')
        logging.debug(f'delta_log_alpha: {delta_log_alpha}, alpha: {alpha}')

        # Accept or reject
        u = np.random.uniform()
        if u < alpha:
            theta_current = theta_proposed.copy()
            acceptance_count += 1
            logging.info(f'Iteration {t}: Proposal accepted.')
        else:
            logging.info(f'Iteration {t}: Proposal rejected.')

        # Store samples after burn-in and thinning
        if t > burn_in and (t - burn_in) % thin == 0:
            samples.append(theta_current.copy())

    acceptance_rate = acceptance_count / num_iterations
    logging.info(f'Acceptance rate: {acceptance_rate}')
    return samples

# Run the MCMC algorithm
samples = metropolis_hastings()

# Extract samples for each parameter
lambda1_samples = [sample['lambda1'] for sample in samples]
lambda2_samples = [sample['lambda2'] for sample in samples]
A_samples = [sample['A'] for sample in samples]
B_samples = [sample['B'] for sample in samples]

# Plot the trace and histogram for each parameter
def plot_results(samples, parameter_name, true_value=None):
    plt.figure(figsize=(12, 5))

    # Trace plot
    plt.subplot(1, 2, 1)
    plt.plot(samples)
    plt.title(f'Trace plot for {parameter_name}')
    plt.xlabel('Iteration')
    plt.ylabel(parameter_name)
    if true_value is not None:
        plt.axhline(y=true_value, color='r', linestyle='--', label='True Value')
        plt.legend()

    # Histogram
    plt.subplot(1, 2, 2)
    plt.hist(samples, bins=30, density=True)
    plt.title(f'Posterior distribution of {parameter_name}')
    plt.xlabel(parameter_name)
    plt.ylabel('Density')
    if true_value is not None:
        plt.axvline(x=true_value, color='r', linestyle='--', label='True Value')
        plt.legend()

    plt.tight_layout()
    plt.show()

# Plot the results
plot_results(lambda1_samples, 'lambda1', true_value=lambda1_truth)
plot_results(lambda2_samples, 'lambda2', true_value=lambda2_truth)
plot_results(A_samples, 'A', true_value=A_truth)
plot_results(B_samples, 'B', true_value=B_truth)
