import numpy as np
from scipy.special import erfc

def mono_conv(self, x, A, lam):
        sigma = self.irf_width/self.step
        
        term1 = (1/2) * A * np.exp((1/2) * lam * (2*self.irf_mean + lam*(sigma**2) - 2*x))
        term2 = lam * erfc((self.irf_mean + lam*(sigma**2) - x)/(sigma*np.sqrt(2)))
        
        return term1*term2

def log_likelihood(params, x, y, model_func):
    A, lam = params
    y_model = model_func(x, A, lam)
    return -0.5 * np.sum((y - y_model)**2)

def proposal(params):
    return params + np.random.normal(0, 0.1, size=params.shape)

def metropolis_hastings(x, y, model_func, initial_params, n_iter=10000):
    current_params = initial_params
    current_log_likelihood = log_likelihood(current_params, x, y, model_func)
    samples = []
    
    for i in range(n_iter):
        proposed_params = proposal(current_params)
        proposed_log_likelihood = log_likelihood(proposed_params, x, y, model_func)
        
        if np.log(np.random.rand()) < (proposed_log_likelihood - current_log_likelihood):
            current_params = proposed_params
            current_log_likelihood = proposed_log_likelihood
            
        samples.append(current_params)
    
    return np.array(samples)

# Example usage
x_data = np.linspace(0, 10, 100)
y_data = mono_conv(x=x_data, A=1.0, lam=0.5)  # Generate some synthetic data

initial_params = np.array([1.0, 0.5])
samples = metropolis_hastings(x_data, y_data, mono_conv, initial_params)

# Analyze the samples to get parameter estimates and uncertainties
