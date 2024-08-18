import numpy as np
import matplotlib.pyplot as plt

def bi_rld(data, g, s):
    D0, D1, D2, D3 = data

    R = D1*D1 - D2*D0
    P = D3*D0 - D2*D1
    Q = D2*D2 - D3*D1
    disc = P**2 - 4*R*Q
    y = (-P + np.sqrt(disc))/(2*R)
    x = (-P - np.sqrt(disc))/(2*R)
    S = s * ((x**2)*D0 - (2*x*D1) + D2)
    T = (1-((x*D1 - D2)/(x*D0 - D1))) ** (g/s)

    tau1 = -s/np.log(y)
    tau2 = -s/np.log(x)

    A1 = (-(x*D0 - D1)**2) * np.log(y) / (S * T) 
    A2 = (-R * np.log(x)) / (S * ((x**(g/s)) - 1))

    return (A1, tau1, A2, tau2)


def get_prob(reg, params, zeta=0.01):
    start, stop = reg
    A1, tau1, A2, tau2 = params

    A = A1/(A1+A2)
    lam1 = 1/tau1
    lam2 = 1/tau2

    prob = zeta * (-A*np.exp(-stop*lam1) -(1-A)*np.exp(-stop*lam2)
                   + A*np.exp(-start*lam1) + (1-A)*np.exp(-start*lam2))
    return prob

params = [1, 5, 2, 20]
regs = [[3, 8], [5, 10], [7, 12], [9, 14]]

probs = []
for reg in regs:
    probs.append(get_prob(reg, params))

bit_success = []
for prob in probs:
    bit_success.append(1-((1-prob)**97))

print(bi_rld(bit_success,5,2))