import matplotlib.pyplot as plt
import numpy as np
from scipy.special import erf, erfc
from scipy.integrate import quad

B = 0.5
lam1 = 1/20
lam2 = 1/5
sigma = 1.5
tau_IRF = 5

def h(t, B, lam1, lam2, sigma, tau_IRF):
        exp1 = np.exp(lam1*(tau_IRF-t) + (lam1**2)*(sigma**2)/2)
        erfc1 = erfc((tau_IRF - t - lam1*(sigma**2))/(sigma*np.sqrt(2)))
        term1 = B*lam1*exp1*erfc1

        exp2 = np.exp(lam2*(tau_IRF-t) + (lam2**2)*(sigma**2)/2)
        erfc2 = erfc((tau_IRF - t - lam2*(sigma**2))/(sigma*np.sqrt(2)))
        term2 = (1-B)*lam2*exp2*erfc2

        print(f'{lam1*(tau_IRF-t) + (lam1**2)*(sigma**2)/2}')

        return term1 + term2

t = np.linspace(0, 50, 400)
plt.plot(t, h(t, B, lam1, lam2, sigma, tau_IRF), label=tau_IRF)
plt.legend()
plt.show()

tau_IRF = 5

for i in range(1, 101):
    res, err = quad(h)