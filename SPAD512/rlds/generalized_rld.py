import numpy as np
import matplotlib.pyplot as plt

def mono_rld(dt, g, t0, data):
    I1, I2 = data

    tau = dt/np.log(I1/I2)

    term1 = tau * np.exp(g/tau - (g+t0)/tau)
    term2 = tau * ((I1/I2)**((-g-t0)/dt))
    A = I1/(term1-term2)

    return (tau, A)

dt = 2
g = 5
t0 = 10.5
data = (2386.51218541, 1953.91091879)

print(mono_rld(dt, g, t0, data))
