import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.special import erfc

class Plotter:
    def __init__(self, config):
        self.config = config

        # compatibility with simulation .json formatting
        if 'integ' not in config:
            self.config['integ'] = self.config['integrations']
        if 'step' not in config:
            self.config['step'] = self.config['gatesteps']
            

    def decay(self, x, amp, tau):
        return amp * np.exp(-x / tau)

    def decay_conv(self, x, A, lam):
        sigma = self.config['irf_width']/self.config['step']
        
        term1 = (1/2) * A * np.exp((1/2) * lam * (2*self.config['irf_mean'] + lam*(sigma**2) - 2*x))
        term2 = lam * erfc((self.config['irf_mean'] + lam*(sigma**2) - x)/(sigma*np.sqrt(2)))
        
        return term1*term2
    
    def plot_all(self, results, filename, show=False):
        A = results['A'].astype(float)
        intensity = results['intensity'].astype(float)
        tau = results['tau'].astype(float)
        print(tau)
        times = results['times'].astype(float)
        full_trace = results['full_trace'].astype(float)
        params = results['params'].astype(float)
        track = int(results['track'])
        
        for i in range(len(tau)):
            for j in range(len(tau[0])):
                if tau[i][j] > 1000:
                    tau[i][j] = 0
                if tau[i][j] < -1000:
                    tau[i][j] = 0

        # for i in range(len(A)):
        #     for j in range(len(A[0])):
        #         if A[i][j] > 10000:
        #             A[i][j] = 0
        #         if A[i][j] < 0:
        #             A[i][j] = 0

        fig, ax = plt.subplots(2, 2, figsize=(7, 7))
        # fig.suptitle(f'{self.config["integ"]*1e-3} ms integ, {self.config["step"]} ns step, {self.config["integ"]*self.config["numsteps"]*1e-3} ms acq time, {self.config["thresh"]} thresh, {track} fits', fontsize=12)
        fig.suptitle('Simulated fit with IRF=N(15, 0.5), 1 ms integ/100 ps step')

        im1 = ax[0, 0].imshow(A, cmap='plasma')
        ax[0, 0].set_title('Amplitudes')
        plt.colorbar(im1, ax=ax[0, 0], label='cts')

        colors = [(1, 0, 0)] + [(i, i, i) for i in np.linspace(0, 1, 255)]
        custom = mcolors.LinearSegmentedColormap.from_list('custom_gray', colors, N=256)
        norm = mcolors.Normalize(vmin=0, vmax=np.max(intensity))
        im2 = ax[0, 1].imshow(intensity, cmap=custom, norm=norm)
        ax[0, 1].set_title('Intensity')
        plt.colorbar(im2, ax=ax[0, 1], label='cts')

        ax[1, 0].set_title('Lifetimes')
        im3 = ax[1, 0].imshow(tau, cmap='hsv')
        plt.colorbar(im3, ax=ax[1, 0], label='ns')
        im3.set_clim(9, 11)

        ax[1, 1].set_title('Fully binned trace')
        ax[1, 1].scatter(times, full_trace, s=5)
        ax[1, 1].plot(times, self.decay_conv(times, params[0], 1/params[1]), label='Fit: tau = {:.2f}'.format(params[1]), color='black')
        ax[1, 1].set_xlabel('Time, ns')
        ax[1, 1].set_ylabel('Counts')
        val = max(full_trace)
        ax[1, 1].set_ylim(0, 1.5 * val)
        ax[1, 1].tick_params(axis='x', which='both', bottom=True, top=True)
        ax[1, 1].tick_params(axis='y', which='both', left=True, right=True)
        ax[1, 1].legend()

        for i, axi in enumerate(ax.ravel()):
            if i != 3:
                axi.set_xticks([])
                axi.set_yticks([])

        plt.tight_layout()
        plt.savefig(filename + '_results.png')

        if show: plt.show()
