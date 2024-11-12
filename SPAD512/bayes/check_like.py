import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erfc, comb
from scipy.integrate import quad

def h(t, B, l1, l2, tau, s):
    term1 = B * l1 * np.exp(l1 * (tau - t) + (l1**2 * s**2) / 2) * erfc((tau - t - l1 * s**2) / (s * np.sqrt(2)))
    term2 = (1 - B) * l2 * np.exp(l2 * (tau - t) + (l2**2 * s**2) / 2) * erfc((tau - t - l2 * s**2) / (s * np.sqrt(2)))
    return term1 + term2

def Pi(i, A, B, l1, l2, tau, s, t0, dt, tw):
    start = t0 + (i - 1) * dt
    end = start + tw
    integral, _ = quad(lambda t: h(t, B, l1, l2, tau, s), start, end)
    return A * integral

def P_chi(chi):
    return 1 - np.exp(-chi)

def logL(y, K, A, B, l1, l2, tau, s, chi, t0, dt, tw):
    Pc = P_chi(chi)
    L = 0
    for i in range(len(y)):
        P = Pi(i + 1, A, B, l1, l2, tau, s, t0, dt, tw)
        Pt = P + Pc
        Pt = np.clip(Pt, 1e-10, 1 - 1e-10)
        L += np.log(comb(K[i], y[i])) + y[i] * np.log(Pt) + (K[i] - y[i]) * np.log(1 - Pt)
    return L

np.random.seed(42)
n = 10
A_true = 0.5
B_true = 0.7
l1_true = 0.2
l2_true = 0.05
tau_true = 2.0
s_true = 0.1
chi_true = 0.02
t0_true = 0.1
dt_true = 0.5
tw_true = 5

K = np.full(n, 100)
counts = []

for i in range(n):
    P = Pi(i + 1, A_true, B_true, l1_true, l2_true, tau_true, s_true, t0_true, dt_true, tw_true)
    Pc = P_chi(chi_true)
    Pt = P + Pc
    c = np.random.binomial(K[i], Pt)
    counts.append(c)

counts = np.array(counts)

ranges = {
    'A': np.linspace(0.1, 1.0, 20),
    'B': np.linspace(0.1, 1.0, 20),
    'l1': np.linspace(0.001, 1.0, 20),
    'l2': np.linspace(0.001, 1.0, 20),
    'chi': np.linspace(0.0001, 0.05, 20)
}

fixed = {
    'A': A_true,
    'B': B_true,
    'l1': l1_true,
    'l2': l2_true,
    'tau': tau_true,
    's': s_true,
    'chi': chi_true,
    't0': t0_true,
    'dt': dt_true,
    'tw': tw_true
}

fig, axes = plt.subplots(2, 3, figsize=(18, 12))

for idx, (name, vals) in enumerate(ranges.items()):
    Ls = []
    for val in vals:
        params = fixed.copy()
        params[name] = val
        L = logL(counts, K, **params)
        Ls.append(L)
    
    ax = axes[idx // 3, idx % 3]
    ax.plot(vals, Ls, marker='o')
    ax.axvline(fixed[name], color='r', linestyle='--')
    ax.set_title(f"{name}")
    ax.grid()

p1, p2 = 'l1', 'l2'
vals1 = np.linspace(0, 0.5, 50)  
vals2 = np.linspace(0, 0.5, 50)
L_matrix = np.zeros((len(vals1), len(vals2)))

for i, v1 in enumerate(vals1):
    for j, v2 in enumerate(vals2):
        params = fixed.copy()
        params[p1] = v1
        params[p2] = v2
        L_matrix[i, j] = logL(counts, K, **params)

ax = axes[1, 2]
X, Y = np.meshgrid(vals2, vals1)
contour = ax.contourf(X, Y, L_matrix, levels=50)
plt.colorbar(contour, ax=ax)

ax.axvline(l2_true, color='r', linestyle='--')
ax.axhline(l1_true, color='r', linestyle='--')

ax.axvline(l1_true, color='g', linestyle='--')
ax.axhline(l2_true, color='g', linestyle='--')

ax.set_title(f"{p1} vs {p2}")
ax.set_xlabel(p2)
ax.set_ylabel(p1)

axes[0,2].axvline(l2_true, color='g', linestyle='--')
axes[1,0].axvline(l1_true, color='g', linestyle='--')
axes[0,1].axvline(1-B_true, color='g', linestyle='--')

plt.tight_layout()
plt.show()