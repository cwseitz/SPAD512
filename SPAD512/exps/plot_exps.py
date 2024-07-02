import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.special import erfc, erfcx
from scipy.ndimage import median_filter

class Plotter:
    def __init__(self, config):
        self.config = config

    def decay(self, x, amp, lam):
        return amp * np.exp(-x * lam)

    def decay_conv(self, x, A, lam):
        term1 = (1/2) * A * np.exp((1/2) * lam * (2*self.config['irf_mean'] + lam*(self.config['irf_width']**2) - 2*x))

        term2 = lam * erfc((self.config['irf_mean'] + lam*(self.config['irf_width']**2) - x)/(self.config['irf_width']*np.sqrt(2)))
        return term1*term2
    
    def decay_double(self, x, amp1, tau1, amp2, tau2):
        return amp1 * np.exp(-x / tau1) + amp2 * np.exp(-x / tau2)

    def bi_conv(self, x, A1, lam1, A2, lam2):
        term1 = (A1*lam1/2) * np.exp((1/2) * lam1 * (2*float(self.config['irf_mean']) + lam1*(self.config['irf_width']**2) - 2*x))
        term2 = erfc((float(self.config['irf_mean']) + lam1*(self.config['irf_width']**2) - x)/(self.config['irf_width']*np.sqrt(2)))
        term3 = (A2*lam2/2) * np.exp((1/2) * lam2 * (2*float(self.config['irf_mean']) + lam2*(self.config['irf_width']**2) - 2*x))
        term4 = erfc((float(self.config['irf_mean']) + lam2*(self.config['irf_width']**2) - x)/(self.config['irf_width']*np.sqrt(2)))

        full = (term1*term2) + (term3*term4)
        return full
    
    def plot_all(self, results, filename, show=False):
        A1 = results['A1'].astype(float)
        A2 = results['A2'].astype(float)
        tau1 = results['tau1'].astype(float)
        tau2 = results['tau2'].astype(float)
        intensity = results['intensity'].astype(float)
        full_trace = results['full_trace'].astype(float)
        full_params = results['full_params'].astype(float)
        track = results['track'].astype(int)
        times = results['times'].astype(float)
        
        for i in range(len(tau1)):
            for j in range(len(tau1[0])):
                if tau1[i][j] > 100:
                    tau1[i][j] = 0
                    A1[i][j] = 0
                if tau1[i][j] < 0:
                    tau1[i][j] = 0
                    A1[i][j] = 0
                if tau2[i][j] > 100:
                    tau2[i][j] = 0
                    A2[i][j] = 0
                if tau2[i][j] < 0:
                    tau2[i][j] = 0
                    A2[i][j] = 0

        for i in range(len(A1)):
            for j in range(len(A1[0])):
                if A1[i][j] > 10000:
                    A1[i][j] = 10000
                if A1[i][j] < 0:
                    A1[i][j] = 0
                if A2[i][j] > 10000:
                    A2[i][j] = 10000
                if A2[i][j] < 0:
                    A2[i][j] = 0
        
        if self.config['fit'] in ('mono', 'mono_conv', 'log_mono_conv', 'mh_mono_conv'):
            fig, ax = plt.subplots(2, 2, figsize=(7, 7))
            fig.suptitle(f'{self.config["integ"]} us integ, {int(self.config["step"])} ps step, {int(self.config["integ"]*self.config["numsteps"]*1e-3)} ms acq time, {self.config["thresh"]} thresh, {track} fits', fontsize=12)
            # fig.suptitle('Guessed IRF=N(10, 0.1), QD image; 1 ms integ/100 ps step')

            im1 = ax[0, 0].imshow(A1, cmap='plasma')
            ax[0, 0].set_title('Amplitudes')
            plt.colorbar(im1, ax=ax[0, 0], label='cts')

            colors = [(1, 0, 0)] + [(i, i, i) for i in np.linspace(0, 1, 255)]
            custom = mcolors.LinearSegmentedColormap.from_list('custom_gray', colors, N=256)
            norm = mcolors.Normalize(vmin=0, vmax=np.max(intensity))
            im2 = ax[0, 1].imshow(intensity, cmap=custom, norm=norm)
            ax[0, 1].set_title('Intensity')
            plt.colorbar(im2, ax=ax[0, 1], label='cts')


            colors = [(0, 0, 0)] + [plt.cm.seismic(i) for i in np.linspace(0, 1, 255)]
            custom2 = mcolors.LinearSegmentedColormap.from_list('custom_seismic', colors, N=256)

            ax[1, 0].set_title('Lifetimes')
            im3 = ax[1, 0].imshow(tau1, cmap=custom2)
            plt.colorbar(im3, ax=ax[1, 0], label='ns')
            im3.set_clim(6, 14)

            ax[1, 1].set_title('Fully binned trace')
            ax[1, 1].scatter(times, full_trace, s=5)
            if self.config['fit'] in ('mono'):
                ax[1, 1].plot(times, self.decay(times, full_params[0], full_params[1]), label='Fit: tau = {:.2f}'.format(1/full_params[1]), color='black')
            elif self.config['fit'] in ('mono_conv', 'log_mono_conv', 'mh_mono_conv'):
                ax[1, 1].plot(times, self.decay_conv(times, full_params[0], full_params[1]), label='Fit: tau = {:.2f}'.format(1/full_params[1]), color='black')
            ax[1, 1].set_xlabel('Time, ns')
            ax[1, 1].set_ylabel('Counts')
            ax[1, 1].set_ylim(0, 1.5 * max(full_trace))
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
        
        if (self.config['fit'] in ('bi', 'bi_conv', 'mh_bi', 'nnls_bi', 'nnls_bi_conv')):
            for i in range(len(A1)):
                for j in range(len(A1[0])):
                    if tau2[i][j] < tau1[i][j]:
                        temp = A1[i][j]
                        A2[i][j] = A1[i][j]
                        A1[i][j] = temp

                        temp = tau1[i][j]
                        tau2[i][j] = tau1[i][j]
                        tau1[i][j] = temp

            fig, ax = plt.subplots(2, 3, figsize=(12, 8))
            fig.suptitle(f'{self.config["integ"]} us integ, {int(self.config["step"])} ps step, {int(self.config["integ"]*self.config["numsteps"]*1e-3)} ms acq time, {self.config["thresh"]} thresh, {track} fits', fontsize=12)
            # fig.suptitle('Simulated fit with IRF=N(15, 0.5), 1 ms integ/100 ps step')

            # A1 = median_filter(A1, size = 3)
            # A2 = median_filter(A2, size = 3)
            # tau1 = median_filter(tau1, size = 3)
            # tau2 = median_filter(tau2, size = 3)

            colors = [(0, 0, 0)] + [plt.cm.plasma(i) for i in np.linspace(0, 1, 255)]
            custom = mcolors.LinearSegmentedColormap.from_list('custom_plasma', colors, N=256)
            im1 = ax[0, 0].imshow(A1, cmap=custom)
            ax[0, 0].set_title('Smaller Amplitude')
            plt.colorbar(im1, ax=ax[0, 0], label='cts')
            im2 = ax[0, 1].imshow(A2, cmap=custom)
            ax[0, 1].set_title('Larger Amplitude')
            plt.colorbar(im2, ax=ax[0, 1], label='cts')

            colors = [(0, 0, 0)] + [plt.cm.seismic(i) for i in np.linspace(0, 1, 255)]
            custom = mcolors.LinearSegmentedColormap.from_list('custom_seismic', colors, N=256)
            # colors2 =  [(0, 0, 0)] + [plt.cm.PiYG(i) for i in np.linspace(0, 1, 255)]
            # custom2 = mcolors.LinearSegmentedColormap.from_list('custom_PiYG', colors2, N=256)
            im3 = ax[1, 0].imshow(tau1, cmap=custom)
            ax[1, 0].set_title('Smaller Lifetime')
            plt.colorbar(im3, ax=ax[1, 0], label='ns')
            im4 = ax[1, 1].imshow(tau2, cmap=custom)
            ax[1, 1].set_title('Larger Lifetime')
            plt.colorbar(im4, ax=ax[1, 1], label='cts')

            colors = [(1, 0, 0)] + [(i, i, i) for i in np.linspace(0, 1, 255)]
            custom = mcolors.LinearSegmentedColormap.from_list('custom_gray', colors, N=256)
            norm = mcolors.Normalize(vmin=0, vmax=np.max(intensity))
            im5 = ax[0, 2].imshow(intensity, cmap=custom, norm=norm)
            ax[0, 2].set_title('Intensity')
            plt.colorbar(im5, ax=ax[0, 2], label='cts')

            ax[1, 2].set_title('Fully binned trace')
            ax[1, 2].scatter(times, full_trace, s=5)
            if self.config['fit'] in ('nnls_bi_conv', 'bi_conv'):
                ax[1, 2].plot(times, self.bi_conv(times, full_params[0], 1/full_params[1], full_params[2], 1/full_params[3]), label='Fit: tau = {:.2f}, {:.2f}'.format(1/full_params[1], 1/full_params[3]), color='black')
            else:
                ax[1, 2].plot(times, self.decay_double(times, full_params[0], 1/full_params[1], full_params[2], 1/full_params[3]), label='Fit: tau = {:.2f}, {:.2f}'.format(1/full_params[1], 1/full_params[3]), color='black')
            print(full_params)
            ax[1, 2].set_xlabel('Time, ns')
            ax[1, 2].set_ylabel('Counts')
            val = max(full_trace)
            ax[1, 2].set_ylim(0, 1.3 * val)
            ax[1, 2].tick_params(axis='x', which='both', bottom=True, top=True)
            ax[1, 2].tick_params(axis='y', which='both', left=True, right=True)
            ax[1, 2].legend()

            for i, axi in enumerate(ax.ravel()):
                if i != 5:
                    axi.set_xticks([])
                    axi.set_yticks([])

            plt.tight_layout()
            plt.savefig(filename + '_results.png')

            if show: plt.show()
