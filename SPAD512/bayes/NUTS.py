import numpy as np
import pymc as pm
import pytensor.tensor as pt
import arviz as az
import matplotlib.pyplot as plt
from scipy.special import erf, erfc

K          = 10_000          # laser pulses per gate
num_steps  = 20              # how many gate positions
step       = 5.0             # ns   (t_s)
offset     = 2.5             # ns   (t_o)
t_w        = 5.0             # ns   (gate width)
tau_irf    = 0.0             # ns   (μ of IRF)
sigma_irf  = 1.4             # ns   (σ of IRF)

def exgauss_cdf(t, mu, sigma, tau):
    """
    normal_cdf(z) - exp(sigma^2/2tau^2 - (t-mu)/tau * normal_cdf(z - sigma/tau) with z = (t-mu)/sigma
    gaussian CDF convolved with an exponential
    """
    z   = (t - mu) / sigma
    z2  = z - sigma / tau
    Phi = 0.5 * (1.0 + pt.erf(z / pt.sqrt(2.0)))
    Phi2 = 0.5 * (1.0 + pt.erf(z2 / pt.sqrt(2.0)))
    return Phi - pt.exp(sigma**2 / (2.0 * tau**2) - (t - mu) / tau) * Phi2

def gate_prob(t0, t1, A, B, tau1, tau2):
    cdf1 = exgauss_cdf(t1, tau_irf, sigma_irf, tau1) - exgauss_cdf(t0, tau_irf, sigma_irf, tau1)
    cdf2 = exgauss_cdf(t1, tau_irf, sigma_irf, tau2) - exgauss_cdf(t0, tau_irf, sigma_irf, tau2)
    return A * (B * cdf1 + (1.0 - B) * cdf2)

def exgauss_cdf_np(t, mu, sigma, tau):
    z   = (t - mu) / sigma
    z2  = z - sigma / tau
    Phi = 0.5 * (1.0 + erf(z))
    Phi2 = 0.5 * (1.0 + erf(z2))
    return Phi - np.exp(sigma**2 / (2.0 * tau**2) - (t - mu) / tau) * Phi2

def gate_prob_np(t0, t1, A, B, tau1, tau2):
    cdf1 = exgauss_cdf_np(t1, tau_irf, sigma_irf, tau1) - exgauss_cdf_np(t0, tau_irf, sigma_irf, tau1)
    cdf2 = exgauss_cdf_np(t1, tau_irf, sigma_irf, tau2) - exgauss_cdf_np(t0, tau_irf, sigma_irf, tau2)
    return A * (B * cdf1 + (1.0 - B) * cdf2)

if __name__ ==  '__main__':
    rng = np.random.default_rng(716743)
    lam1_true, lam2_true = 0.05, 0.20     
    tau1_true, tau2_true = 1/lam1_true, 1/lam2_true
    A_true = 0.1
    B_true = 0.8
    chi_true = 1e-3

    t_start = offset + np.arange(num_steps) * step
    t_end   = t_start + t_w



    P_chi_true = 1 - np.exp(-chi_true)
    P_gate     = gate_prob_np(t_start, t_end, A_true, B_true, tau1_true, tau2_true)
    P_tot_true = P_gate + P_chi_true
    y_obs      = rng.binomial(K, P_tot_true).astype("int64")

    with pm.Model() as model:
        tau1 = pm.LogNormal("tau1", mu=np.log(20), sigma=1.0)
        tau2 = pm.LogNormal("tau2", mu=np.log(5), sigma=1.0)
        lam1 = pm.Deterministic("lam1", 1.0 / tau1)
        lam2 = pm.Deterministic("lam2", 1.0 / tau2)

        A = pm.Beta("A", alpha=3, beta=18)       
        B = pm.Beta("B", alpha=5, beta=2)                  

        t0 = pt.as_tensor_variable(t_start)
        t1 = pt.as_tensor_variable(t_end)

        cdf1 = exgauss_cdf(t1, tau_irf, sigma_irf, tau1) - \
            exgauss_cdf(t0, tau_irf, sigma_irf, tau1)
        cdf2 = exgauss_cdf(t1, tau_irf, sigma_irf, tau2) - \
            exgauss_cdf(t0, tau_irf, sigma_irf, tau2)

        Pgate = A * (B * cdf1 + (1.0 - B) * cdf2)
        Ptot  = pm.Deterministic("P_tot", pt.clip(Pgate, 1e-8, 1 - 1e-8))

        y = pm.Binomial("y", n=K, p=Ptot, observed=y_obs)

        with model:
            idata = pm.sample(5000, tune=5000, target_accept=0.95, initvals={"tau1": 20, "tau2": 5, "A": 0.1, "B": 0.8})

    az.summary(idata, var_names=["tau1","tau2","A","B"], round_to=4)

    az.plot_trace(idata, var_names=["lam1","lam2","A","B"])
    plt.show()
