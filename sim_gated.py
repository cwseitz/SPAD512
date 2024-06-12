import numpy as np
import flimlib
import time
import scipy.optimize as opt
import matplotlib.pyplot as plt
from multiprocessing import Pool

''' 
Simulation of exponential fitting of fluorescent lifetime imaging data acquired by a time-gated SPAD
To Do:
- Test limits of multiprocessing in this context
- Optimize data generation
- Code data visualization
- Run full (finer) integration/step simulation series
'''

'''Simulation Parameters'''
freq_sim = 10  # laser frequency in MHz
integ_sims = [0.001, 0.01, 0.1, 1, 10]  # integration times to simulate with in ms
step_sims = [0.018, 0.05, 0.09, 0.18, 0.5, 0.9, 1.8, 5]  # gate step sizes in ns
offset_sim = 0.018  # gate offset in ns
tau_sim = 10  # ground truth lifetime to simulate
width_sim = 5  # gate width in ns
iter = 1000 # number of times to iterate for each combination of integrations and steps

'''Generates data given a ground truth lifetime and SPAD acquisition parameters'''
def genData(freq, integ, step, offset, width, tau):
    numgates = int(1e3 * freq * integ)
    interpulse = 1e3 / freq
    numsteps = int(interpulse / step) # take floor to prevent pulse overhang
    lam = 1/tau 

    data = np.zeros(numsteps, dtype=int)
    for i in range(numsteps):
        prob = np.exp(-lam * (offset + i*step)) - np.exp(-lam * (offset + i*step + width))  # direct integration of PDF
        draws = np.random.rand(numgates) < prob
        count = np.sum(draws)
        data[i] += count

    return data

'''Analyzes data from genData using scipy.optimize.curve_fit (LMA algorithm) on a mono-exponential fit'''
def getLifetime(data, step):
    def decay(x, amp, tau):
        return amp * np.exp(-x / tau)

    def fit_decay(data, step):
        x = np.arange(len(data)) * step
        y = data

        initial_guess = [np.max(data), 2.0] # use max for the amplitude, choice is ultimately insignificant with LMA algorithm

        try: 
            params, cov = opt.curve_fit(decay, x, y, p0=initial_guess)
        except RuntimeError:
            params = [0, 0] # not really sure how else to handle curve_fit failing after many iterations
        
        return params

    LMA_params = fit_decay(data, step)
    return LMA_params

'''Inner function for multiprocessing'''
def runSimulation(freq, integ, step, offset, width, tau):
    data = genData(freq, integ, step, offset, width, tau)
    info = getLifetime(data, step)
    return info[1] # returns lifetime (tau) in nanoseconds

'''Outer function for multiprocessing'''
def job(n): # need to call n as a parameter for multiprocessing syntax
    out = runSimulation(freq_sim, integ_sim, step_sim, offset_sim, width_sim, tau_sim) 
    return out

'''Data display function (placeholder until I decide how to actually visualize data)'''
def display(im):
    plt.imshow(im)

'''Main code'''
stdevs = np.zeros((len(integ_sims), len(step_sims)))
for i, integ_sim in enumerate(integ_sims):
    for j, step_sim in enumerate(step_sims):
        tic = time.time()
        print(f'Beginning simulation for {integ_sim} ms integration and {step_sim} ns step size.')

        if __name__ == '__main__': # multiprocessing bit
            with Pool(100) as p:
                lifetimes = p.map(job, range(iter))

        stdevs[i][j] += np.std(lifetimes) # store standard deviation over all iterations as summary stat

        toc = time.time()
        print(f'Simulation complete in {toc-tic} seconds. Lifetime stdev {stdevs[i][j]}\n')
