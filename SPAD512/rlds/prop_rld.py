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

def gen(reg, params, numgates=1000000, iter=10000, zeta=0.01, kernel=25):
    start, stop = reg
    A1, tau1, A2, tau2 = params

    A = A1/(A1+A2)
    lam1 = 1/tau1
    lam2 = 1/tau2

    prob = zeta * (-A*np.exp(-stop*lam1) -(1-A)*np.exp(-stop*lam2)
                   + A*np.exp(-start*lam1) + (1-A)*np.exp(-start*lam2))

    counts = 0
    for i in range(kernel):
        counts += np.random.binomial(numgates, prob, size=iter)
    return counts

def hist_single(off, g, s, params):
    off = 0
    g = 5
    s = 2.5
    params = [1, 5, 2, 20]
    gates = [(off + i * s, off + g + i * s) for i in range(4)]
    data = []
    for i, gate in enumerate(gates):
        data.append(gen(gate, params))

    n_A1, n_tau1, n_A2, n_tau2 = bi_rld(data, g, s)

    plt.hist(n_tau1, bins=50,  density=True)
    print(n_tau1)
    plt.title(f'Distribution of end values for {g} ns gate length and {s} ns step size')
    plt.xlabel('Tau 1')
    plt.xlim(0, 100)
    plt.ylabel('PDF')
    plt.show()

def cmap_all(off, g_sims, s_sims, params, thresh=0.2):
    successes = np.zeros((len(g_sims), len(s_sims), 2))
    for j, g in enumerate(g_sims):
        for k, s in enumerate(s_sims):
            gates = [(off + i * s, off + g + i * s) for i in range(4)]
            data = []
            for i, gate in enumerate(gates):
                data.append(gen(gate, params))

            n_A1, n_tau1, n_A2, n_tau2 = bi_rld(data, g, s)
            successes[j,k,0] += np.sum((n_tau1 >= params[1]-thresh*params[1]) & (n_tau1 <= params[1]+thresh*params[1])) / (len(data[0]))
            successes[j,k,1] += np.sum((n_tau2 >= params[3]-thresh*params[3]) & (n_tau2 <= params[3]+thresh*params[3])) / (len(data[0]))

    fig, ax = plt.subplots(1,2)   
    plt.suptitle(f'Proportion of samples with lifetimes within {100*thresh} % error')

    cax1 = ax[0].imshow(successes[:,:,0])
    ax[0].set_ylabel('Gate length (ns)')
    ax[0].set_xlabel('Step size (ns)')
    ax[0].set_yticks(np.linspace(0, len(g_sims), num=len(g_sims), endpoint=False))
    ax[0].set_yticklabels(g_sims)
    ax[0].set_xticks(np.linspace(0, len(s_sims), num=len(s_sims), endpoint=False))
    ax[0].set_xticklabels(s_sims)
    fig.colorbar(cax1)

    cax2 = ax[1].imshow(successes[:,:,1])
    ax[1].set_ylabel('Gate length (ns)')
    ax[1].set_xlabel('Step size (ns)')
    ax[1].set_yticks(np.linspace(0, len(g_sims), num=len(g_sims), endpoint=False))
    ax[1].set_yticklabels(g_sims)
    ax[1].set_xticks(np.linspace(0, len(s_sims), num=len(s_sims), endpoint=False))
    ax[1].set_xticklabels(s_sims)
    fig.colorbar(cax2)

    plt.show()

off = 0
g_sims = [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
s_sims = [15, 16, 17, 18, 19, 20, 21, 22, 23, 24]
params = [1, 5, 2, 20]
thresh = 0.2

hist_single(off, g_sims, s_sims, params)