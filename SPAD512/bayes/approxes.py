import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erf, erfc

# '''abram stegun approximation'''
# def erfc_approx(x):
#     b1 = 0.254829592
#     b2 = -0.284496736
#     b3 = 1.421413741
#     b4 = -1.453152027
#     b5 = 1.061405429
#     p = 0.3275911

#     t = 1 / (1 + p * np.abs(x))
#     poly = b1 * t + b2 * (t**2) + b3 * (t**3) + b4 * (t**4) + b5 * (t**5)
#     z = np.exp(-x**2)
#     result = z * poly
#     result = np.where(x >= 0, result, 2 - result)
    
#     return result


# x = np.linspace(-5, 5, 1000) 
# abram = erfc_approx(x)
# scipy = erfc(x)

# err = np.abs(abram - scipy)


# plt.figure(figsize=(12, 6))

# plt.subplot(1, 2, 1)
# plt.plot(x, abram, label='Abramowitz-Stegun', color='blue')
# plt.plot(x, scipy, label='SciPy', color='orange', linestyle='--')
# plt.title('Comparison of erfcs')
# plt.xlabel('x')
# plt.ylabel('erfc(x)')
# plt.legend()
# plt.grid()

# plt.subplot(1, 2, 2)
# plt.plot(x, err, label='Absolute Error', color='red')
# plt.title('Absolute Error scipy - AS')
# plt.xlabel('x')
# plt.ylabel('Absolute Error')
# plt.yscale('log')  
# plt.legend()
# plt.grid()

# plt.tight_layout()
# plt.show()

'''log combinatoric approximation'''
import numpy as np
import scipy.special as sp
import matplotlib.pyplot as plt

K = 10000
y_values = np.arange(1, K, 10)  

exact = [np.log(sp.comb(K, y)) for y in y_values]
approx = [sp.gammaln(K + 1) - sp.gammaln(y + 1) - sp.gammaln(K - y + 1) for y in y_values]

abs_err = np.abs(np.array(exact) - np.array(approx))
rel_err = abs_err / np.abs(np.array(exact))

plt.plot(y_values, abs_err, label="Absolute Error")
plt.plot(y_values, rel_err, label="Relative Error")
plt.legend()
plt.show()

for y, exact, approx, abs_err, rel_err in zip(
    y_values[:10], exact[:10], approx[:10], abs_err[:10], rel_err[:10]
):
    print(f"y={y:5d}, Exact={exact:.5e}, Approx={approx:.5e}, Abs Error={abs_err:.5e}, Rel Error={rel_err:.5e}")

