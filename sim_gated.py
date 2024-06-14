import numpy as np
import time
import scipy.optimize as opt
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from mpl_toolkits.axes_grid1 import make_axes_locatable
from multiprocessing import Pool
from skimage.io import imsave, imread

''' 
Simulation of exponential fitting of fluorescent lifetime imaging data acquired by a time-gated SPAD
'''

'''Simulation Parameters'''
freq_sim = 10  # laser frequency in MHz
integ_sims = [0.001, 0.0025, 0.005, 0.0075, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 0.75, 1]
step_sims = [0.018, 0.04, 0.06, 0.1, 0.15, 0.2, 0.25, 0.5, 0.75, 1, 1.5, 2, 2.5, 5]
offset_sim = 0.018  # gate offset in ns
tau_sim = [1, 5, 10]  # ground truth lifetime to simulate
width_sim = 5  # gate width in nsW
iter = 1000 # number of times to iterate for each combination of integrations and steps
zeta_sim = 0.05 # probability of detection per pulse, from QY, detector eff., cross section, excitation flux

'''Generates data given a ground truth lifetime and SPAD acquisition parameters'''
def genData(freq, integ, step, offset, width, tau, zeta):
    numgates = int(1e3 * freq * integ)
    interpulse = 1e3 / freq
    numsteps = int(interpulse / step) # take floor to prevent pulse overhang
    lam = 1/tau 

    data = np.zeros(numsteps, dtype=int)
    for i in range(numsteps):
        prob = np.exp(-lam * (offset + i*step)) - np.exp(-lam * (offset + i*step + width))  # direct integration of PDF
        prob = zeta * prob # prob is sim'd for single pulse, so can directly multiply by zeta
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

    # plots decay curve from a particular iteration
    # x = np.arange(len(data)) * step
    # plt.figure(figsize=(6, 4))
    # plt.plot(x, data, 'bo', markersize=3, label='Data')
    # plt.plot(x, decay(x, *LMA_params), 'r--', label='Fit: tau = {:.2f}'.format(LMA_params[1]))
    # plt.xlabel('Time, ns')
    # plt.ylabel('Counts')
    # plt.legend()
    # plt.title('Simulated Decay for 10 ms integration, 5 ns step')
    # plt.show()

    return LMA_params

'''Inner function for multiprocessing'''
def runSimulation(freq, integ, step, offset, width, tau):
    data = genData(freq, integ, step, offset, width, tau)
    info = getLifetime(data, step)
    return info[1] # returns lifetime (tau) in nanoseconds

'''Outer function for multiprocessing'''
def job(n): # need to call n as a parameter for multiprocessing syntax
    out = runSimulation(freq_sim, integ_sim, step_sim, offset_sim, width_sim, tau_sim, zeta_sim) 
    return out

'''Data display function'''
def display(mean_image, std_image, integs, steps, tau, show = False):
    xlen, ylen = np.shape(mean_image)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    
    # plot mean
    norm_mean = mcolors.TwoSlopeNorm(vmin=np.min(mean_image), vcenter=tau, vmax=np.max(mean_image))
    cax1 = ax1.imshow(mean_image, cmap='PRGn', norm=norm_mean)
    cbar1 = fig.colorbar(cax1, ax=ax1, shrink = 0.6)
    cbar1.set_label('Mean of Lifetimes, ns')
    ax1.set_title('Mean of Lifetimes')
    ax1.set_xlabel('Step size (ns)')
    ax1.set_ylabel('Integration time (ms)')
    ax1.set_yticks(np.linspace(0, xlen, num=xlen, endpoint=False))
    ax1.set_yticklabels(integs)
    ax1.set_xticks(np.linspace(0, ylen, num=ylen, endpoint=False))
    ax1.set_xticklabels(steps)
    plt.setp(ax1.get_xticklabels(), rotation=30)

    # plot stdevs
    norm_std = mcolors.Normalize(vmin=0, vmax=1, clip=True)
    cax2 = ax2.imshow(std_image, cmap='hot', norm=norm_std)
    cbar2 = fig.colorbar(cax2, ax=ax2, shrink = 0.6)
    cbar2.set_label('Standard Deviation of Lifetimes, ns')
    ax2.set_title('Standard Deviation of Lifetimes')
    ax2.set_xlabel('Step size (ns)')
    ax2.set_ylabel('Integration time (ms)')
    ax2.set_yticks(np.linspace(0, xlen, num=xlen, endpoint=False))
    ax2.set_yticklabels(integs)
    ax2.set_xticks(np.linspace(0, ylen, num=ylen, endpoint=False))
    ax2.set_xticklabels(steps)
    plt.setp(ax2.get_xticklabels(), rotation=30)

    plt.savefig(f'{tau}ns_full', bbox_inches='tight')

    if show:
        plt.tight_layout()
        plt.show()


'''Main code'''
tic1 = time.time()
stdevs = np.zeros((len(integ_sims), len(step_sims)))
means = np.zeros((len(integ_sims), len(step_sims)))
for tau_sim in tau_sims:
    for i, integ_sim in enumerate(integ_sims):
        for j, step_sim in enumerate(step_sims):
            tic = time.time()
            print(f'Beginning simulation for {integ_sim} ms integration and {step_sim} ns step size.')

            if __name__ == '__main__': # multiprocessing bit
                with Pool(100) as p:
                    lifetimes = p.map(job, range(iter))

            # box plots lifetimes for a particular integ/step combo
            # plt.figure(figsize=(6, 4))
            # plt.boxplot(lifetimes, vert=False,  patch_artist=True)
            # plt.title('Box Plot of Lifetimes for 1 ms integration, 5 ns step')
            # plt.xlabel('Lifetimes, in ns')
            # plt.show()

            stdevs[i][j] += np.std(lifetimes) 
            means[i][j] += np.average(lifetimes) 

            toc = time.time()
            print(f'Simulation ns complete in {toc-tic} seconds. Lifetime stdev {stdevs[i][j]}\n')
    tic2 = time.time()
    print(f'Full simulation for {tau_sim} ns lifetime complete in {tic2-tic1} seconds.')
    imsave(f'{tau_sim}ns_stdevs.tif',stdevs)
    imsave(f'{tau_sim}ns_means.tif', means)
    display(means, stdevs, integ_sims, step_sims, tau_sim, show = False)

# stdevs = imread('sim4_stdevs.tif')
# means = imread('sim4_means.tif',)

display(means, stdevs, integ_sims, step_sims, tau_sim, show = False)
