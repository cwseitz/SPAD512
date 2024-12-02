import numpy as np
import matplotlib.pyplot as plt
import scipy.special as sp
from scipy.integrate import quad

start = 5
end = 10
tau_irf = 1.5
sigma_irf = 0.5
B = 0.5
lam1 = 0.05
lam2 = 0.2

def h(t, tau_irf, sigma_irf, B, lam1, lam2):
    term1 = B * lam1 * np.exp(lam1 * (tau_irf - t) + 0.5 * lam1**2 * sigma_irf**2) \
            * sp.erfc((tau_irf - t - lam1 * sigma_irf**2) / (sigma_irf * np.sqrt(2)))
    term2 = (1 - B) * lam2 * np.exp(lam2 * (tau_irf - t) + 0.5 * lam2**2 * sigma_irf**2) \
            * sp.erfc((tau_irf - t - lam2 * sigma_irf**2) / (sigma_irf * np.sqrt(2)))
    return term1 + term2

norm, norm_err = quad(h, 0, 1000, args=(tau_irf, sigma_irf, B, lam1, lam2))  
res, err = quad(h, start, end, args=(tau_irf, sigma_irf, B, lam1, lam2))
print(res/norm)