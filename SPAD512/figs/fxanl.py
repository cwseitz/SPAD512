import numpy as np
import matplotlib.pyplot as plt

def f(x, s):
    num = s * (x**(s-1)) * (1-x)
    den = 1 - (x**s)
    return np.sqrt(num/den)

x = np.linspace(0, 1, 400)  
s = np.linspace(1, 100, 400) 

X, S = np.meshgrid(x, s)

F = f(X, S)

plt.figure(figsize=(10, 6))
plt.contourf(X, S, F, levels=50, cmap='viridis')
plt.colorbar(label='f(x, s)')
plt.title('Heat Map of f(x, s)')
plt.xlabel('x')
plt.ylabel('s')
plt.show()

F_max = np.max(F)
F_max
