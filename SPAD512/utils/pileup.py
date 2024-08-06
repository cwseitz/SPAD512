import numpy as np
import matplotlib.pyplot as plt

# test with contiguous RLD
interpulse = 100 # interpulse time in NS
tau = 20 # ground truth monoexponential lifetime
zeta = 0.05 # probability of photon detection
step = 10 # gate step size (and width for contiguous case)
start = 0 # first gate opening time

def perturb(interpulse, tau, step, start):
    return 0

def gen(tau, zeta, step, start):
    lam = 1/tau
    D0 = zeta * (np.exp(-lam * (start)) - np.exp(-lam * (start + step)))
    D1 = zeta * (np.exp(-lam * (start + step)) - np.exp(-lam * (start + 2*step)))
    return D0, D1

def rld(D0, D1, step):
    A = (D0**2) * (np.log(D0/D1)) / (step*(D0-D1)) 
    tau = step / (np.log(D0/D1))
    return A, tau




