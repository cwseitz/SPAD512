import pymc as pm
import numpy as np
import arviz as az
import scipy.special as sp
import matplotlib.pyplot as plt
from pytensor.graph import Apply, Op
import pytensor.tensor as pt
import pytensor
from scipy.optimize import approx_fprime
rng = np.random.default_rng(716743)

'''model constants (known in experiment)
ground truth values are defined in gen() to avoid variable naming annoyances
'''
K = 10000 # number of laser pulses per step
numsteps = 20 # number of steps
step = 5 # ns
offset = 0.018 # ns
width = 5 # ns
tau_irf = 1.5
sigma_irf = 0.5

A_true = 0.5
B_true = 0.5
lam1_true = 0.05
lam2_true = 0.2
chi_true = 0.0001

'''define helper functions for likelihood'''
def h(t, tau_irf, sigma_irf, B, lam1, lam2):
    term1 = B * lam1 * np.exp(lam1 * (tau_irf - t) + 0.5 * lam1**2 * sigma_irf**2) \
            * sp.erfc((tau_irf - t - lam1 * sigma_irf**2) / (sigma_irf * np.sqrt(2)))
    term2 = (1 - B) * lam2 * np.exp(lam2 * (tau_irf - t) + 0.5 * lam2**2 * sigma_irf**2) \
            * sp.erfc((tau_irf - t - lam2 * sigma_irf**2) / (sigma_irf * np.sqrt(2)))
    return term1 + term2

def dh_dB(t, tau_irf, sigma_irf, B, lam1, lam2):
    term1 = 1 * lam1 * np.exp(lam1 * (tau_irf - t) + 0.5 * lam1**2 * sigma_irf**2) \
            * sp.erfc((tau_irf - t - lam1 * sigma_irf**2) / (sigma_irf * np.sqrt(2)))
    term2 = -1 * lam2 * np.exp(lam2 * (tau_irf - t) + 0.5 * lam2**2 * sigma_irf**2) \
            * sp.erfc((tau_irf - t - lam2 * sigma_irf**2) / (sigma_irf * np.sqrt(2)))
    return term1 + term2

def dh_dlam(t, tau_irf, sigma_irf, B, lam1, lam2):
    outer = B * np.exp((lam1*(sigma_irf**2) - 2*t + 2*tau_irf)/2)
    term1 = lam1 * sigma_irf * np.sqrt(2/np.pi) * np.exp((-(lam1 * sigma_irf**2 + t - tau_irf)**2)/(2 * sigma_irf**2))
    term2 = (1 + (lam1**2)*(sigma_irf**2) + lam1*(tau_irf - t)) \
            * sp.erfc(-(lam1 * sigma_irf**2 + t - tau_irf)/(np.sqrt(2)*sigma_irf))
    return outer * (term1 + term2)

def P_i(start, end, A, B, lam1, lam2, tau_irf, sigma_irf, h_func):
    t_vals = np.linspace(start, end, 100)  # for trapezioidal sum, quad integration probably not needed
    h_vals = h_func(t_vals, tau_irf, sigma_irf, B, lam1, lam2)
    return A * np.trapz(h_vals, t_vals)

def gen(K, numsteps, step, offset, width, tau_irf, sigma_irf, A=0.3, B=0.5, lam1=0.05, lam2=0.2, chi=0.0001):
    P_chi = 1 - np.exp(-chi)
    data = []

    for i in range(numsteps):
        start = offset + i*step
        end = start + width

        Pi = P_i(start, end, A, B, lam1, lam2, tau_irf, sigma_irf, h)

        P_tot = Pi + P_chi

        data.append(np.random.binomial(K, P_tot))

    return data


data = gen(K, numsteps, step, offset, width, tau_irf, sigma_irf, A_true, B_true, lam1_true, lam2_true, chi_true)
data = np.float64(data)

def cal_loglike(lam1, lam2, A, B, chi, data):
    loglike = np.zeros(len(data))
    print(f'lam1: {lam1, lam2, A, B, chi}')
    print(f'data: {data}')

    for i, yi in enumerate(data):
        Pi = P_i(offset + i*step, offset + i*step + width,
            A, B, lam1, lam2, tau_irf, sigma_irf, h
        )
        
        Pchi = 1 - np.exp(-chi)
        Ptot = Pi + Pchi

        log_binom = sp.gammaln(K + 1) - sp.gammaln(yi + 1) - sp.gammaln(K - yi + 1)
        loglike[i] += log_binom

        print(yi)
        loglike[i] += yi * np.log(Ptot)
        loglike[i] += (K-yi) * np.log(1-Ptot)

    return loglike     

class LogLike(Op):
    def make_node(self, lam1, lam2, A, B, chi, data) -> Apply:
        lam1 = pt.as_tensor_variable(lam1)
        lam2 = pt.as_tensor_variable(lam2)
        A = pt.as_tensor_variable(A)
        B = pt.as_tensor_variable(B)
        chi = pt.as_tensor_variable(chi)
        data = pt.as_tensor_variable(data)

        inputs = [lam1, lam2, A, B, chi, data]
        outputs = [data.type()]

        return Apply(self, inputs, outputs)
    
    def perform(self, node: Apply, inputs: list[np.ndarray], outputs: list[list[None]]) -> None:
        lam1, lam2, A, B, chi, data = inputs  
        loglike_eval = cal_loglike(lam1, lam2, A, B, chi, data)
        outputs[0][0] = np.asarray(loglike_eval)

loglike_op = LogLike()
test_out = loglike_op(lam1_true, lam2_true, A_true, B_true, chi_true, data)
print(f'test out eval: {test_out.eval()}')
print(f'not in blackbox: {cal_loglike(lam1_true, lam2_true, A_true, B_true, chi_true, data)}')
pytensor.dprint(test_out, print_type=True)


if __name__ == '__main__':
    # 1: generate data
    data = gen(K, numsteps, step, offset, width, tau_irf, sigma_irf) # use kwargs to change ground truth

    # 2: custom likelihood function
    def custom_dist_loglike(data, lam1, lam2, A, B, chi):
        return loglike_op(lam1, lam2, A, B, chi, data)

    with pm.Model() as model:
        # 3: define priors and likelihood
        lam1 = pm.Uniform("lam1", lower=0, upper=2)
        lam2 = pm.Uniform("c", lower=0, upper=2)
        A = pm.Beta("A", alpha=1, beta=1)
        B = pm.Beta("B", alpha=1, beta=1)
        chi = pm.Uniform("chi", lower=0.00005, upper=0.00015)

        likelihood = pm.CustomDist(
            "likelihood", lam1, lam2, A, B, chi, observed=data, logp=custom_dist_loglike
        )

        ip = model.initial_point()
        model.debug(verbose=True)
        # idata = pm.sample(3000, tune=1000)
       

        # az.plot_trace(idata, lines=[("lam1", {}, 0.05), ("lam2", {}, 0.2)])
        # plt.show()