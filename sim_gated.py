import numpy as np
import time
import scipy.optimize as opt
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from multiprocessing import Pool
from skimage.io import imsave, imread
from scipy.io import savemat

''' 
Simulation of exponential fitting of fluorescent lifetime imaging data acquired by a time-gated SPAD
'''

'''Manipulated Simulation Parameters'''
# integ_sims = [0.001, 0.0025, 0.005, 0.0075, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 0.75, 1, 5, 10] # integration times in ms
# step_sims = [0.018, 0.04, 0.06, 0.1, 0.15, 0.2, 0.25, 0.5, 0.75, 1, 1.5, 2, 2.5, 5] # gate step sizes (not width) in ns
# tau_sims = [10, 5, 1]  # ground truth lifetimes in ns
integ_sims = [1]
step_sims = [0.09]
tau_sims = [10]

'''Standard Simulation Parameters'''
freq_sim = 10  # laser frequency in MHz
offset_sim = 0.018  # gate offset in ns
width_sim = 5  # gate width in ns
iter = 1 # number of times to iterate for each combination of integrations and steps
zeta_sim = 0.01 # probability of detection per pulse, from QY, detector eff., cross section, excitation flux

'''Generates data given a ground truth lifetime and SPAD acquisition parameters'''
def genData(freq, integ, step, offset, width, tau, zeta):
    numgates = int(1e3 * freq * integ)
    interpulse = 1e3 / freq
    numsteps = int(interpulse / step)
    lam = 1 / tau

    data = np.zeros(numsteps, dtype=int)
    steps = np.arange(numsteps) * step
    prob = zeta * (np.exp(-lam * (offset + steps)) - np.exp(-lam * (offset + steps + width)))
    
    for i in range(numsteps):
        draws = np.random.rand(numgates) < prob[i]
        data[i] = np.sum(draws)

    savemat('BNP-LA-main/my_array.mat', {'my_array': data})

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
            success = True
        except RuntimeError:
            params = [100, 100] # not really sure how else to handle curve_fit failing after many iterations
            success = False
        
        return params, success

    LMA_params, success = fit_decay(data, step)

    # plots decay curve from a particular iteration
    x = np.arange(len(data)) * step
    plt.figure(figsize=(6, 4))
    plt.plot(x, data, 'bo', markersize=3, label='Data')
    plt.plot(x, decay(x, *LMA_params), 'r--', label='Fit: tau = {:.2f}'.format(LMA_params[1]))
    plt.xlabel('Time, ns')
    plt.ylabel('Counts')
    plt.legend()
    plt.title('Simulated Decay for 1 ms integration, 90 ps step')
    plt.show()

    return LMA_params, success

'''Inner function for multiprocessing'''
def runSimulation(freq, integ, step, offset, width, tau, zeta):
    data = genData(freq, integ, step, offset, width, tau, zeta)
    if max(data) > 0:
        info, success = getLifetime(data, step)
        return info[1], success  # returns lifetime (tau) in nanoseconds and success flag
    return 0, False

'''Outer function for multiprocessing'''
def job(n): # need to call n as a parameter for multiprocessing syntax
    out, success = runSimulation(freq_sim, integ_sim, step_sim, offset_sim, width_sim, tau_sim, zeta_sim)
    return out, success

'''Data display function'''
def display(mean_image, std_image, integs, steps, tau, show = True):
    xlen, ylen = np.shape(mean_image)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    
    # plot mean
    mean_image = mean_image - 5
    imsave(str(tau_sim) + 'ns_means.tif',mean_image)
    norm = mcolors.TwoSlopeNorm(vmin=np.min(mean_image), vcenter=tau, vmax=np.max(mean_image))
    cax1 = ax1.imshow(mean_image, cmap='seismic', norm=norm)
    cbar1 = fig.colorbar(cax1, ax=ax1, shrink = 0.6)
    cbar1.set_label('Means, ns')
    ax1.set_title('Mean Lifetimes')
    ax1.set_xlabel('Step size (ns)')
    ax1.set_ylabel('Integration time (ms)')
    ax1.set_yticks(np.linspace(0, xlen, num=xlen, endpoint=False))
    ax1.set_yticklabels(integs)
    ax1.set_xticks(np.linspace(0, ylen, num=ylen, endpoint=False))
    ax1.set_xticklabels(steps)
    plt.setp(ax1.get_xticklabels(), rotation=45)

    # plot stdevs
    std_image = np.clip(std_image, -1, 2)
    norm = mcolors.TwoSlopeNorm(vmin=-1, vcenter=0, vmax=2)
    cax2 = ax2.imshow(std_image, cmap='seismic', norm=norm)
    cbar2 = fig.colorbar(cax2, ax=ax2, shrink = 0.6)
    cbar2.set_label('St Devs, ns')
    ax2.set_title('Standard Deviation of Lifetimes')
    ax2.set_xlabel('Step size (ns)')
    ax2.set_ylabel('Integration time (ms)')
    ax2.set_yticks(np.linspace(0, xlen, num=xlen, endpoint=False))
    ax2.set_yticklabels(integs)
    ax2.set_xticks(np.linspace(0, ylen, num=ylen, endpoint=False))
    ax2.set_xticklabels(steps)
    plt.setp(ax2.get_xticklabels(), rotation=45)

    plt.savefig(str(tau) + 'ns_full', bbox_inches='tight')
    print('Figure saved as ' + str(tau) + 'ns_full.png')

    if show:
        plt.tight_layout()
        plt.show()


'''Main code'''
for tau_sim in tau_sims:
    tic1 = time.time()
    stdevs = np.zeros((len(integ_sims), len(step_sims)))
    means = np.zeros((len(integ_sims), len(step_sims)))
    for i, integ_sim in enumerate(integ_sims):
        for j, step_sim in enumerate(step_sims):
            tic = time.time()
            print(f'Beginning simulation for {integ_sim} ms integ, {step_sim} ns step, {tau_sim} ns lifetime.')

            if __name__ == '__main__': # multiprocessing bit
                with Pool(100) as p:
                    results = p.map(job, range(iter))

            lifetimes = [res[0] for res in results if res[1]]  # Filter out non-converging fits

            # box plots lifetimes for a particular integ/step combo
            # plt.figure(figsize=(6, 4))
            # plt.boxplot(lifetimes, vert=False,  patch_artist=True)
            # plt.title('Box Plot of Lifetimes for 10 ms integration, 100 ps step')
            # plt.xlabel('Lifetimes, in ns')
            # plt.show()

            if ((len(lifetimes) > 0) and (max(lifetimes) > 0)):  # Ensure there are valid lifetimes
                stdevs[i][j] += np.std(lifetimes) 
                means[i][j] += np.average(lifetimes) 
            else: 
                stdevs[i][j] = -1
                means[i][j] = -1
            toc = time.time()
            print(f'Simulation complete in {toc-tic} seconds. Lifetime stdev {stdevs[i][j]}, mean {means[i][j]}\n')
    tic2 = time.time()
    print(f'Full simulation for {tau_sim} ns lifetime complete in {tic2-tic1} seconds.')
    imsave(str(tau_sim) + 'ns_stdevs.tif',stdevs)
    imsave(str(tau_sim) + 'ns_means.tif',means)
    display(means, stdevs, integ_sims, step_sims, tau_sim, show = False)

# stdevs = imread('1ns_stdevs.tif')
# means = imread('1ns_means.tif',)

# display(means, stdevs, integ_sims, step_sims, tau_sim, show = True)
