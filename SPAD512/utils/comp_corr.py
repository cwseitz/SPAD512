import numpy as np
import matplotlib.pyplot as plt

def mine(x, K, Imax):
    term1 = (1 - x/Imax)**(Imax/K)
    return K * (1 - term1)

def spad512(x, Imax):
    return -Imax*np.log(1 - x/Imax)

x = np.linspace(1, 254, 500)
Imax = 255
K = 1000
plt.plot(x, mine(x, K, Imax), label='New correction')
plt.plot(x, spad512(x, Imax), label='Existing correction')
plt.legend()
plt.show()  