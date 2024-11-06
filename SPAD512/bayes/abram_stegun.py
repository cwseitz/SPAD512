import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erf, erfc

# Custom erfc function implementation
def erfc_approx(x):
    b1 = 0.254829592
    b2 = -0.284496736
    b3 = 1.421413741
    b4 = -1.453152027
    b5 = 1.061405429
    p = 0.3275911

    # Calculate t and poly values for positive |x|
    t = 1 / (1 + p * np.abs(x))
    poly = b1 * t + b2 * (t**2) + b3 * (t**3) + b4 * (t**4) + b5 * (t**5)
    z = np.exp(-x**2)
    
    # Calculate final result
    result = z * poly

    # Adjust for negative x values
    result = np.where(x >= 0, result, 2 - result)
    
    return result


x = np.linspace(-5, 5, 1000) 
abram = erfc_approx(x)
scipy = erfc(x)

err = np.abs(abram - scipy)


plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.plot(x, abram, label='Abramowitz-Stegun', color='blue')
plt.plot(x, scipy, label='SciPy', color='orange', linestyle='--')
plt.title('Comparison of erfcs')
plt.xlabel('x')
plt.ylabel('erfc(x)')
plt.legend()
plt.grid()

plt.subplot(1, 2, 2)
plt.plot(x, err, label='Absolute Error', color='red')
plt.title('Absolute Error scipy - AS')
plt.xlabel('x')
plt.ylabel('Absolute Error')
plt.yscale('log')  
plt.legend()
plt.grid()

plt.tight_layout()
plt.show()