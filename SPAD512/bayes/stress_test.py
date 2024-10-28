import numpy as np
import time
import matplotlib.pyplot as plt
from scipy.integrate import quad
from scipy.special import erfc

B = 0.5
lam1 = 1 / 20
lam2 = 1 / 5
sigma = 1.5
tau_IRF = 5

def h(t, B, lam1, lam2, sigma, tau_IRF):
    exp1 = np.exp(lam1 * (tau_IRF - t) + (lam1 ** 2) * (sigma ** 2) / 2)
    erfc1 = erfc((tau_IRF - t - lam1 * (sigma ** 2)) / (sigma * np.sqrt(2)))
    term1 = B * lam1 * exp1 * erfc1

    exp2 = np.exp(lam2 * (tau_IRF - t) + (lam2 ** 2) * (sigma ** 2) / 2)
    erfc2 = erfc((tau_IRF - t - lam2 * (sigma ** 2)) / (sigma * np.sqrt(2)))
    term2 = (1 - B) * lam2 * exp2 * erfc2

    return term1 + term2

iterations = 100  

t_mins = np.linspace(0.018, 100, 1000)
t_maxes = np.linspace(10.018, 100.018, 1000)

start_time = time.time()
for j in range(iterations):
    mid_time = time.time()

    for i in range(len(t_mins)):    
        result, error = quad(h, t_mins[i], t_maxes[i], args=(B, lam1, lam2, sigma, tau_IRF))

    print(f'iteration {j} complete in {time.time() - mid_time}')
end_time = time.time()

time_taken = end_time - start_time
print(f"Time taken for {iterations} iterations with {len(t_mins)} steps: {time_taken:.2f} seconds")

'''estimated per gate integration time is 0.3 ms'''

# t_values = np.linspace(t_min, t_max, 400)
# integrated_values = [quad(h, t_min, t, args=(B, lam1, lam2, sigma, tau_IRF))[0] for t in t_values]
# plt.plot(t_values, integrated_values, label=f"Integrated h(t) over [{t_min}, t]")
# plt.legend()
# plt.show()
