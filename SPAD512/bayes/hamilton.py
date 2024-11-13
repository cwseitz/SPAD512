import pymc as pm
import numpy as np
import arviz as az
import scipy.special as sp
import matplotlib.pyplot as plt
from pytensor.graph import Apply, Op
import pytensor as pt
from scipy.optimize import approx_fprime

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
    print(f'A: {A}, B: {B}, lam1: {lam1}, lam2: {lam2}')
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


# data = gen(K, numsteps, step, offset, width, tau_irf, sigma_irf, A, B, lam1, lam2, chi)


def cal_loglike(lam1, lam2, A, B, chi, data):
    loglike = 0

    for i, yi in enumerate(data):
        Pi = P_i(offset + i*step, offset + i*step + width,
            A, B, lam1, lam2, tau_irf, sigma_irf, h
        )
        Pchi = 1 - np.exp(-chi)
        Ptot = Pi + Pchi

        loglike += np.log(sp.comb(K, yi))
        loglike += yi * np.log(Ptot)
        loglike += (K-yi) * np.log(1-Ptot)
        print(f'Pi: {Pi}')

    return loglike     

def grad_main(lam1, lam2, A, B, chi, data):
    grad_wrt_lam1 = 0
    grad_wrt_lam2 = 0
    grad_wrt_A = 0
    grad_wrt_B = 0
    grad_wrt_chi = 0

    for i, val in enumerate(data): # iterate over each step individually for now
        # ptot calculation
        Pi = P_i(offset + i*step, offset + i*step + width,
            A, B, lam1, lam2, tau_irf, sigma_irf, h
        )
        Pchi = 1 - np.exp(-chi)
        Ptot = Pi + Pchi

        # chi
        dP_dchi = np.exp(-chi)
        grad_wrt_chi += ((val/Ptot) + ((val - K)/(1-Ptot))) * dP_dchi

        # A
        dP_dA = (Ptot - 1 + np.exp(-chi))/A
        grad_wrt_A += ((val/Ptot) + ((val - K)/(1-Ptot))) * dP_dA

        # B
        t_vals = np.linspace(offset + i*step,  offset + i*step + width, 100)  # for trapezioidal sum, quad integration probably not needed
        h_vals = dh_dB(t_vals, tau_irf, sigma_irf, B, lam1, lam2)
        dP_dB = A * np.trapz(h_vals, t_vals)
        grad_wrt_B += ((val/Ptot) + ((val - K)/(1-Ptot))) * dP_dB

        # lam1
        h_vals = dh_dlam(t_vals, tau_irf, sigma_irf, B, lam1, lam2)
        dP_dlam1 = A * np.trapz(h_vals, t_vals)
        grad_wrt_lam1 += ((val/Ptot) + ((val - K)/(1-Ptot))) * dP_dlam1

        # lam2
        h_vals = dh_dlam(t_vals, tau_irf, sigma_irf, B, lam2, lam1) # just swap lam1/lam2 because there is no lam1 lam2 mixing
        dP_dlam2 = A * np.trapz(h_vals, t_vals)
        grad_wrt_lam2 += ((val/Ptot) + ((val - K)/(1-Ptot))) * dP_dlam2
    
    return grad_wrt_lam1, grad_wrt_lam2, grad_wrt_A, grad_wrt_B, grad_wrt_chi

class LogLikeWithGrad(Op):
    def make_node(self, lam1, lam2, A, B, chi, data) -> Apply:
        # same as before
        lam1 = pt.tensor.as_tensor_variable(lam1)
        lam2 = pt.tensor.as_tensor_variable(lam2)
        A = pt.tensor.as_tensor_variable(A)
        B = pt.tensor.as_tensor_variable(B)
        chi = pt.tensor.as_tensor_variable(chi)
        data = pt.tensor.as_tensor_variable(data)

        inputs = [lam1, lam2, A, B, chi, data]
        outputs = [data.type()]
        return Apply(self, inputs, outputs)

    def perform(self, node: Apply, inputs: list[np.ndarray], outputs: list[list[None]]) -> None:
        # also same
        lam1, lam2, A, B, chi, data = inputs  
        loglike_eval = cal_loglike(lam1, lam2, A, B, chi, data)
        outputs[0][0] = np.asarray(loglike_eval)

    def grad(self, inputs: list[pt.tensor.TensorVariable], g: list[pt.tensor.TensorVariable]) -> list[pt.tensor.TensorVariable]:
        lam1, lam2, A, B, chi, data = inputs
        if lam1.type.ndim != 0 or lam2.type.ndim != 0 or A.type.ndim != 0 or B.type.ndim != 0 or chi.type.ndim != 0:
            raise NotImplementedError("Graident only implemented for scalars")
        
        grad_wrt_lam1, grad_wrt_lam2, grad_wrt_A, grad_wrt_B, grad_wrt_chi = loglikegrad_op(lam1, lam2, A, B, chi, data)

        # out_grad is a tensor of gradients of the Op outputs wrt to the function cost
        [out_grad] = g
        return [
            pt.tensor.sum(out_grad * grad_wrt_lam1),
            pt.tensor.sum(out_grad * grad_wrt_lam2),
            pt.tensor.sum(out_grad * grad_wrt_A),
            pt.tensor.sum(out_grad * grad_wrt_B),
            pt.tensor.sum(out_grad * grad_wrt_chi),
            pt.gradient.grad_not_implemented(self, 4, data), # maybe don't need with data??
        ]

class LogLikeGrad(Op):
    def make_node(self, m, c, sigma, x, data) -> Apply:
        lam1 = pt.tensor.as_tensor_variable(lam1)
        lam2 = pt.tensor.as_tensor_variable(lam2)
        A = pt.tensor.as_tensor_variable(A)
        B = pt.tensor.as_tensor_variable(B)
        chi = pt.tensor.as_tensor_variable(chi)
        data = pt.tensor.as_tensor_variable(data)

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

if __name__ == '__main__':
    # 1: generate data
    data = gen(K, numsteps, step, offset, width, tau_irf, sigma_irf) # use kwargs to change ground truth

    # 2: custom likelihood function
    def custom_dist_loglike(data, lam1, lam2, A, B, chi):
        return loglikewithgrad_op(lam1, lam2, A, B, chi, data)

    with pm.Model() as grad_model:
        # 3: define priors and likelihood
        lam1 = pm.Uniform("lam1", lower=0, upper=2)
        lam2 = pm.Uniform("c", lower=0, upper=2)
        A = pm.Beta("A", alpha=1, beta=1)
        B = pm.Beta("B", alpha=1, beta=1)
        chi = pm.Gamma("chi", alpha=2, beta=0.001)

        likelihood = pm.CustomDist(
            "likelihood", lam1, lam2, A, B, chi, observed=data, logp=custom_dist_loglike
        )

        # perform sampling
        with grad_model:
            idata_grad = pm.sample()

        az.plot_trace(idata_grad, lines=[("lam1", {}, 0.05), ("lam2", {}, 0.2)])
        plt.show()