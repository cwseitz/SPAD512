import pymc as pm
import numpy as np
import arviz as az
import scipy.special as sp
import matplotlib.pyplot as plt
from pytensor.graph import Apply, Op
import pytensor.tensor as pt
import pytensor
from scipy.optimize import approx_fprime
from scipy.integrate import quad
import os
rng = np.random.default_rng(716743)

'''model constants (known in experiment)
ground truth values are defined in gen() to avoid variable naming annoyances
'''
K = 10000 # number of laser pulses per step
numsteps = 20 # number of steps
step = 5 # ns
offset = 2.5 # ns
width = 5 # ns
tau_irf = 0
sigma_irf = 1.4

A_true = 0.1
B_true = 0.8
lam1_true = 0.05
lam2_true = 0.2
chi_true = 0.001

def h(t, tau_irf, sigma_irf, B, lam1, lam2):
    term1 = B * lam1 * np.exp(lam1 * (tau_irf - t) + 0.5 * lam1**2 * sigma_irf**2) \
            * sp.erfc((tau_irf - t - lam1 * sigma_irf**2) / (sigma_irf * np.sqrt(2)))
    term2 = (1 - B) * lam2 * np.exp(lam2 * (tau_irf - t) + 0.5 * lam2**2 * sigma_irf**2) \
            * sp.erfc((tau_irf - t - lam2 * sigma_irf**2) / (sigma_irf * np.sqrt(2)))
    return term1 + term2

def P_i(start, end, A, B, lam1, lam2, tau_irf, sigma_irf, h_func):
    norm, norm_err = quad(h_func, 0, 1000, args=(tau_irf, sigma_irf, B, lam1, lam2))  
    res, err = quad(h_func, start, end, args=(tau_irf, sigma_irf, B, lam1, lam2))
    return A * res/norm

def gen(K, numsteps, step, offset, width, tau_irf, sigma_irf, lam1=0.05, lam2=0.2, A=0.5, B=0.5, chi=0.0001):
    P_chi = 1 - np.exp(-chi)
    data = []

    for i in range(numsteps):
        start = offset + i*step
        end = start + width

        Pi = P_i(start, end, A, B, lam1, lam2, tau_irf, sigma_irf, h)

        P_tot = Pi + P_chi

        data.append(np.random.binomial(K, P_tot))

    return data

def cal_loglike(lam1, lam2, A, B, chi, data):
    loglike = np.zeros(len(data))

    for i, yi in enumerate(data):
        Pi = P_i(offset + i*step, offset + i*step + width,
            A, B, lam1, lam2, tau_irf, sigma_irf, h
        )
        
        Pchi = 1 - np.exp(-chi)
        Ptot = Pi + Pchi
        Ptot = Ptot.item()

        if Ptot <= 0: Ptot = 1e-8
        if Ptot >= 1: Ptot = 1 - 1e-8

        log_binom = sp.gammaln(K + 1) - sp.gammaln(yi + 1) - sp.gammaln(K - yi + 1)
        loglike[i] += log_binom

        loglike[i] += yi * np.log(Ptot)
        loglike[i] += (K-yi) * np.log(1-Ptot)

    return loglike  



data = gen(K, numsteps, step, offset, width, tau_irf, sigma_irf, 
           A=A_true, B=B_true, lam1=lam1_true, lam2=lam2_true, chi=chi_true)
data = np.float64(data)



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
def custom_dist_loglike(data, lam1, lam2, A, B, chi):
    return loglike_op(lam1, lam2, A, B, chi, data)

# test_out = loglike_op(lam1_true, lam2_true, A_true, B_true, chi_true, data)
# pytensor.dprint(test_out)

if __name__ == '__main__':
    with pm.Model() as model:
        # note these priors down, particularly B prior uniform 0-1 or 0.5-1 has a very large impact
        # lam and chi can take gamma priors, A and B beta priors
        lam1 = pm.Uniform("lam1", lower=0, upper=1)
        lam2 = pm.Uniform("lam2", lower=0, upper=1)
        A = pm.Uniform("A", lower=0, upper=1)
        B = pm.Uniform("B", lower=0.5, upper=1)
        chi = pm.Uniform("chi", lower=0.00005, upper=0.015)
        

        likelihood = pm.CustomDist(
            "likelihood", lam1, lam2, A, B, chi, observed=data, logp=custom_dist_loglike
        )
        with model:
            idata = pm.sample()

        stats = az.summary(idata, round_to=5)
        stats.to_csv("C:\\Users\\ishaa\\Documents\\FLIM\\242211\\summary.csv")
        print(f"Summary statistics saved")

        plots_dir = "C:\\Users\\ishaa\\Documents\\FLIM\\242211"
        os.makedirs(plots_dir, exist_ok=True)

        try:
            az.plot_trace(idata, lines=[("lam1", {}, lam1_true), ("lam2", {}, lam2_true)])
            plt.tight_layout()
            plt.show()
            plt.close()
            print("Trace plot saved.")
        except Exception as e: 
            pass
 
        # try:
        #     az.plot_energy(idata)
        #     plt.tight_layout()
        #     plt.show()
        #     plt.savefig(plots_dir + "\\energy_plot.png")
        #     plt.close()
        #     print("Energy plot saved.")
        # except Exception as e:
        #     pass

        try:
            az.plot_posterior(idata)
            plt.tight_layout()
            plt.show()
            plt.close()
            print("Posterior plot saved.")
        except Exception as e:
            pass

        try:
            az.plot_pair(idata, kind='kde', divergences=True)
            plt.tight_layout()
            plt.show()
            plt.close()
            print("Pairwise scatterplot saved.")
        except Exception as e:
            pass

        try:
            az.plot_rank(idata)
            plt.tight_layout()
            plt.show()
            plt.close()
            print("Rank plot saved.")
        except Exception as e:
            pass

        print(f"plots saved in {plots_dir}.")