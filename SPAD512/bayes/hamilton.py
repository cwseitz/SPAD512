import pymc as pm
import numpy as np
import arviz as az
import scipy.special as sp
import matplotlib.pyplot as plt
from pytensor.graph import Apply, Op
import pytensor as pt
from scipy.optimize import approx_fprime

'''ground truth values (not known)'''
lam1 = 1/20
lam2 = 1/5
A = 0.1
B = 0.5
chi = 1e-4

'''model constants (known in experiment)'''
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

def P_i(start, end, A, B, lam1, lam2, tau_irf, sigma_irf):
    t_vals = np.linspace(start, end, 100)  # for trapezioidal sum, quad integration probably not needed
    h_vals = h(t_vals, tau_irf, sigma_irf, B, lam1, lam2)
    return A * np.trapz(h_vals, t_vals)



'''generate perfect data'''
def gen(K, numsteps, step, offset, width, tau_irf, sigma_irf, A, B, lam1, lam2, chi):
    P_chi = 1 - np.exp(-chi)
    data = []

    for i in range(numsteps):
        start = offset + i*step
        end = start + width

        t_vals = np.linspace(start, end, 100)
        h_vals = h(t_vals, tau_irf, sigma_irf, B, lam1, lam2)
        P_i = A * np.trapz(h_vals, t_vals)

        P_tot = P_i + P_chi

        data.append(np.random.binomial(K, P_tot))

    return data


# data = gen(K, numsteps, step, offset, width, tau_irf, sigma_irf, A, B, lam1, lam2, chi)


'''working through pymc's tutorial on blackboxing'''
def my_model(m, c, x):
    return m * x + c

def my_loglike(m, c, sigma, x, data):
    model = my_model(m, c, x)
    return -0.5 * ((data - model) / sigma) ** 2 - np.log(np.sqrt(2 * np.pi)) - np.log(sigma)

def finite_differences_loglike(m, c, sigma, x, data, eps=1e-7):
    def inner_func(mc, sigma, x, data):
        return my_loglike(*mc, sigma, x, data)
    
    grad_wrt_mc = approx_fprime([m, c], inner_func, [eps, eps], sigma, x, data)
    
    return grad_wrt_mc[:,0], grad_wrt_mc[:,1]

class LogLikeWithGrad(Op):
    def make_node(self, m, c, sigma, x, data) -> Apply:
        # same as before
        m = pt.tensor.as_tensor_variable(m)
        c = pt.tensor.as_tensor_variable(c)
        sigma = pt.tensor.as_tensor_variable(sigma)
        x = pt.tensor.as_tensor_variable(x)
        data = pt.tensor.as_tensor_variable(data)

        inputs = [m, c, sigma, x, data]
        outputs = [data.type()]
        return Apply(self, inputs, outputs)

    def perform(self, node: Apply, inputs: list[np.ndarray], outputs: list[list[None]]) -> None:
        # also same
        m, c, sigma, x, data = inputs  
        loglike_eval = my_loglike(m, c, sigma, x, data)
        outputs[0][0] = np.asarray(loglike_eval)

    def grad(self, inputs: list[pt.tensor.TensorVariable], g: list[pt.tensor.TensorVariable]
    ) -> list[pt.tensor.TensorVariable]:
        # this will return the vector jacobian product for gradient
        m, c, sigma, x, data = inputs
        if m.type.ndim != 0 or c.type.ndim != 0:
            raise NotImplementedError("Graident only implemented for scalar m and c")
        
        grad_wrt_m, grad_wrt_c = loglikegrad_op(m, c, sigma, x, data)

        # out_grad is a tensor of gradients of the Op outputs wrt to the function cost
        [out_grad] = g
        return [
            pt.tensor.sum(out_grad * grad_wrt_m),
            pt.tensor.sum(out_grad * grad_wrt_c),
            # the other 3 inputs still need a gradient but we dont need it because they are model constants
            pt.gradient.grad_not_implemented(self, 2, sigma), # THIS DOESNT WORK AAAAAAAAAAAAAAAAAAAA
            pt.gradient.grad_not_implemented(self, 3, x), # THIS DOESNT WORK AAAAAAAAAAAAAAAAAAAA
            pt.gradient.grad_not_implemented(self, 4, data), # THIS DOESNT WORK AAAAAAAAAAAAAAAAAAAA
        ]

class LogLikeGrad(Op):
    def make_node(self, m, c, sigma, x, data) -> Apply:
        m = pt.tensor.as_tensor_variable(m)
        c = pt.tensor.as_tensor_variable(c)
        sigma = pt.tensor.as_tensor_variable(sigma)
        x = pt.tensor.as_tensor_variable(x)
        data = pt.tensor.as_tensor_variable(data)

        inputs = [m, c, sigma, x, data]
        outputs = [data.type(), data.type()]

        return Apply(self, inputs, outputs)

    def perform(self, node: Apply, inputs: list[np.ndarray], outputs: list[list[None]]) -> None:
        m, c, sigma, x, data = inputs

        grad_wrt_m, grad_wrt_c = finite_differences_loglike(m, c, sigma, x, data)

        outputs[0][0] = grad_wrt_m
        outputs[1][0] = grad_wrt_c

loglikewithgrad_op = LogLikeWithGrad()
loglikegrad_op = LogLikeGrad()

if __name__ == '__main__':
    N = 10 
    sigma = 1
    x = np.linspace(0, 9, N)
    mtrue = 0.4
    ctrue = 3
    truemodel = my_model(mtrue, ctrue, x)

    rng = np.random.default_rng(716743)
    data = sigma * rng.normal(size = N) + truemodel

    # print(finite_differences_loglike(mtrue, ctrue, sigma, x, data))

    # dont mess with order of parameters
    def custom_dist_loglike(data, m, c, sigma, x):
        return loglikewithgrad_op(m, c, sigma, x, data)
    
    m_symb = pt.tensor.scalar("m")
    c_symb = pt.tensor.scalar("c")

    # Get the log-likelihood output and gradient
    logp_output = loglikewithgrad_op(m_symb, c_symb, sigma, x, data)
    grad_wrt_m, grad_wrt_c = pt.gradient.grad(logp_output.sum(), wrt=[m_symb, c_symb])

    print("Gradient with respect to m:", grad_wrt_m.eval({m_symb: mtrue, c_symb: ctrue}))
    print("Gradient with respect to c:", grad_wrt_c.eval({m_symb: mtrue, c_symb: ctrue}))

    with pm.Model() as grad_model:
        m = pm.Uniform("m", lower=-10.0, upper=10.0)
        c = pm.Uniform("c", lower=-10.0, upper=10.0)

        # use custom distribution to implement custom likelihood
        likelihood = pm.CustomDist(
            "likelihood", m, c, sigma, x, observed=data, logp=custom_dist_loglike
        )

        with grad_model:
            idata_grad = pm.sample(step=pm.NUTS())

        az.plot_trace(idata_grad, lines=[("m", {}, mtrue), ("c", {}, ctrue)])
        plt.show()