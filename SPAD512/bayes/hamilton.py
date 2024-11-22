import pymc as pm
import numpy as np
import arviz as az
import scipy.special as sp
import matplotlib.pyplot as plt
from pytensor.graph import Apply, Op
import pytensor.tensor as pt
import pytensor
from scipy.optimize import approx_fprime
import os

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

A_true = 0.1
B_true = 0.8
lam1_true = 0.03
lam2_true = 0.45
chi_true = 0.001

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


data = gen(K, numsteps, step, offset, width, tau_irf, sigma_irf, A=A_true, B=B_true, lam1=lam1_true, lam2=lam2_true, chi=chi_true)


def cal_loglike(lam1, lam2, A, B, chi, data):
    loglike = 0

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

def grad_main(lam1, lam2, A, B, chi, data):
    grad_wrt_lam1 = np.zeros(len(data))
    grad_wrt_lam2 = np.zeros(len(data))
    grad_wrt_A = np.zeros(len(data))
    grad_wrt_B = np.zeros(len(data))
    grad_wrt_chi = np.zeros(len(data))

    for i, val in enumerate(data):
        Pi = P_i(offset + i*step, offset + i*step + width,
            A, B, lam1, lam2, tau_irf, sigma_irf, h
        )
        Pchi = 1 - np.exp(-chi)
        Ptot = Pi + Pchi
        Ptot = Ptot.item()

        if Ptot <= 0: Ptot = 1e-8
        if Ptot >= 1: Ptot = 1 - 1e-8

        # chi
        dP_dchi = np.exp(-chi)
        grad_wrt_chi[i] += ((val / Ptot) - (K - val) / (1 - Ptot)) * dP_dchi

        # A
        dP_dA = Pi / A
        grad_wrt_A[i] += ((val / Ptot) - (K - val) / (1 - Ptot)) * dP_dA

        # B
        t_vals = np.linspace(offset + i*step, offset + i*step + width, 100)
        h_vals = dh_dB(t_vals, tau_irf, sigma_irf, B, lam1, lam2)
        dP_dB = A * np.trapz(h_vals, t_vals)
        grad_wrt_B[i] += ((val / Ptot) - (K - val) / (1 - Ptot)) * dP_dB

        # lam1
        h_vals = dh_dlam(t_vals, tau_irf, sigma_irf, B, lam1, lam2)
        dP_dlam1 = A * np.trapz(h_vals, t_vals)
        grad_wrt_lam1[i] += ((val / Ptot) - (K - val) / (1 - Ptot)) * dP_dlam1

        # lam2
        h_vals = dh_dlam(t_vals, tau_irf, sigma_irf, B, lam2, lam1)  # swap lam1/lam2 for second derivative
        dP_dlam2 = A * np.trapz(h_vals, t_vals)
        grad_wrt_lam2[i] += ((val / Ptot) - (K - val) / (1 - Ptot)) * dP_dlam2
    
    print(f'lam1 grad akjdfhaiugiqGFAIURALIUHFLIAUEWFLI: f{grad_wrt_lam1}')

    return (
        np.array([grad_wrt_lam1]),
        np.array([grad_wrt_lam2]),
        np.array([grad_wrt_A]),
        np.array([grad_wrt_B]),
        np.array([grad_wrt_chi]),
    )

class LogLikeWithGrad(Op):
    def make_node(self, lam1, lam2, A, B, chi, data) -> Apply:
        # same as before
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
        # also same
        lam1, lam2, A, B, chi, data = inputs  
        loglike_eval = cal_loglike(lam1, lam2, A, B, chi, data)
        outputs[0][0] = np.asarray(loglike_eval)

    def grad(self, inputs: list[pt.TensorVariable], g: list[pt.TensorVariable]) -> list[pt.TensorVariable]:
        lam1, lam2, A, B, chi, data = inputs
        grad_wrt_lam1, grad_wrt_lam2, grad_wrt_A, grad_wrt_B, grad_wrt_chi = loglikegrad_op(lam1, lam2, A, B, chi, data)

        # out_grad is a tensor of gradients of the Op outputs wrt to the function cost
        [out_grad] = g
        return [
            pt.sum(out_grad * grad_wrt_lam1),
            pt.sum(out_grad * grad_wrt_lam2),
            pt.sum(out_grad * grad_wrt_A),
            pt.sum(out_grad * grad_wrt_B),
            pt.sum(out_grad * grad_wrt_chi),
            pytensor.gradient.grad_not_implemented(self, 5, data), # maybe don't need with data??
        ]

class LogLikeGrad(Op):
    def make_node(self, lam1, lam2, A, B, chi, data) -> Apply:
        lam1 = pt.as_tensor_variable(lam1)
        lam2 = pt.as_tensor_variable(lam2)
        A = pt.as_tensor_variable(A)
        B = pt.as_tensor_variable(B)
        chi = pt.as_tensor_variable(chi)
        data = pt.as_tensor_variable(data)

        inputs = [lam1, lam2, A, B, chi, data]
        outputs = [data.type(), data.type(), data.type(), data.type(), data.type()]

        return Apply(self, inputs, outputs)

    def perform(self, node: Apply, inputs: list[np.ndarray], outputs: list[list[None]]) -> None:
        lam1, lam2, A, B, chi, data = inputs

        grad_wrt_lam1, grad_wrt_lam2, grad_wrt_A, grad_wrt_B, grad_wrt_chi = grad_main(lam1, lam2, A, B, chi, data)

        outputs[0][0] = grad_wrt_lam1
        outputs[1][0] = grad_wrt_lam2
        outputs[2][0] = grad_wrt_A
        outputs[3][0] = grad_wrt_B
        outputs[4][0] = grad_wrt_chi

loglikewithgrad_op = LogLikeWithGrad()
loglikegrad_op = LogLikeGrad()
def custom_dist_loglike(data, lam1, lam2, A, B, chi):
    return loglikewithgrad_op(lam1, lam2, A, B, chi, data)

if __name__ == '__main__':
    with pm.Model() as grad_model:
        lam1 = pm.Uniform("lam1", lower=0, upper=1)
        lam2 = pm.Uniform("lam2", lower=0, upper=1)
        A = pm.Uniform("A", lower=0, upper=1)
        B = pm.Uniform("B", lower=0.5, upper=1)
        chi = pm.Uniform("chi", lower=0.00005, upper=0.0015)

        likelihood = pm.CustomDist(
            "likelihood", lam1, lam2, A, B, chi, observed=data, logp=custom_dist_loglike
        )

        # Perform sampling
        with grad_model:
            idata_grad = pm.sample()

        # Save summary statistics
        summary_stats = az.summary(idata_grad, round_to=2)
        summary_file = "summary_statistics.csv"
        summary_stats.to_csv(summary_file)
        print(f"Summary statistics saved to {summary_file}")

        plots_dir = "C:\\Users\\ishaa\\Documents\\FLIM\\bayesian"
        os.makedirs(plots_dir, exist_ok=True)

        try:
            az.plot_trace(idata_grad, lines=[("lam1", {}, lam1_true), ("lam2", {}, lam2_true)])
            plt.savefig(os.path.join(plots_dir, "trace_plot.png"))
            plt.close()
            print("Trace plot saved.")
        except Exception as e:
            print(f"Error generating trace plot: {e}")

        try:
            az.plot_energy(idata_grad)
            plt.savefig(os.path.join(plots_dir, "energy_plot.png"))
            plt.close()
            print("Energy plot saved.")
        except Exception as e:
            print(f"Error generating energy plot: {e}")

        try:
            az.plot_posterior(idata_grad)
            plt.savefig(os.path.join(plots_dir, "posterior_plot.png"))
            plt.close()
            print("Posterior plot saved.")
        except Exception as e:
            print(f"Error generating posterior plot: {e}")

        try:
            az.plot_pair(idata_grad, kind='kde', divergences=True)
            plt.savefig(os.path.join(plots_dir, "pair_plot.png"))
            plt.close()
            print("Pairwise scatterplot saved.")
        except Exception as e:
            print(f"Error generating pairwise scatterplot: {e}")

        try:
            az.plot_rank(idata_grad)
            plt.savefig(os.path.join(plots_dir, "rank_plot.png"))
            plt.close()
            print("Rank plot saved.")
        except Exception as e:
            print(f"Error generating rank plot: {e}")

        print(f"All plots saved in {plots_dir}.")