import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.special import erfc
from scipy.ndimage import median_filter

class Plotter:
    def __init__(self, config, **kwargs):
        defaults = {
            'integ': 0,
            'step': 0,
            'numsteps': 0,
            'offset': 0,
            'thresh': 0,
            'fit': "",
            'irf_width': 0,
            'irf_mean': 0,
            'width': 0,
            'filename': "",
            'kernel_size': 0
        }
        defaults.update(config)
        defaults.update(kwargs)

        for key, val in defaults.items():
            setattr(self, key, val)

        self.step *= 1e-3  # ps --> ns
        self.width *= 1e-3
        self.offset *= 1e-3

    def mono(self, x, amp, lam):
        return amp * np.exp(-x * lam)

    def bi(self, x, A, lam1, B, lam2):
        return A * (B * np.exp(-x * lam1) + (1 - B) * np.exp(-x * lam2))



    '''helper'''
    def preprocess_results(self, results):
        A = results['A'].astype(float)
        B = results['B'].astype(float)
        tau1 = results['tau1'].astype(float)
        tau2 = results['tau2'].astype(float)
        intensity = results['intensity'].astype(float)
        full_trace = results['full_trace'].astype(float)
        full_params = results['full_params'].astype(float)
        track = results['track'].astype(int)
        times = results['times'].astype(float)

        k = self.kernel_size
        if k > 0:
            A = A[k:-k, k:-k]
            B = B[k:-k, k:-k]
            tau1 = tau1[k:-k, k:-k]
            tau2 = tau2[k:-k, k:-k]
            intensity = intensity[k:-k, k:-k]

        mask = (tau1 > 100) | (tau1 < 0) | (tau2 > 100) | (tau2 < 0)
        tau1[mask], tau2[mask], A[mask], B[mask] = 0, 0, 0, 0 
        A = np.clip(A, 0, 10000)
        B = np.clip(B, 0, 1)

        return A, B, tau1, tau2, intensity, full_trace, full_params, track, times



    '''generalized plotter functions'''
    def plot_image(self, ax, data, title, cbar_label, cmap, norm=None):
        if isinstance(cmap, tuple):  # Check if cmap is a tuple containing (cmap, norm)
            cmap, norm = cmap
        im = ax.imshow(data, cmap=cmap, norm=norm)
        ax.set_title(title)
        plt.colorbar(im, ax=ax, label=cbar_label)

    def plot_trace(self, ax, times, full_trace, full_params, fit_type):
        ax.set_title('Fully binned trace')
        
        # full_trace /= np.max(full_trace)
        ax.scatter(times, full_trace, s=5)
        if fit_type == 'mono':
            ax.plot(times, self.mono(times, full_params[0], full_params[1]), label=f'Fit: tau = {(1 / full_params[1]):.2f} ns', color='black')
        elif fit_type == 'bi':
            ax.plot(times, self.bi(times, full_params[0] , full_params[1], full_params[2], full_params[3]), label=f'Fit: tau1 = {(1/full_params[1]):.2f} ns, tau2 = {(1/full_params[3]):.2f} ns', color='black')
        elif fit_type == 'mono_rld':
            ax.plot(times, self.mono(times, full_params[0], full_params[1]), label=f'Fit: tau = {(1/full_params[1]):.2f} ns', color='black')
        
        ax.set_xlabel('Time, ns')
        ax.set_ylabel('Counts')
        ax.set_ylim(0, 1.5 * np.max(full_trace))
        ax.tick_params(axis='x', which='both', bottom=True, top=True)
        ax.tick_params(axis='y', which='both', left=True, right=True)
        ax.legend()

    def plot_rld(self, ax, times, full_trace, full_params, fit_type):
        ax.set_title('RLD Visualization')
        times = np.linspace(0, 100, 1000)
        counts = 0.5*np.exp(-times/20) + 0.5*np.exp(-times/5)

        regs = []
        for i in range(4):
            regs.append((self.offset + i*self.step, self.offset + i*self.step + self.width))
        cols = ['red', 'yellow', 'green', 'blue']
    
        ax.plot(times, counts)
        for (start, end), col in zip(regs, cols):
            mask = (times >= start) & (times <= end)
            ax.fill_between(times[mask], counts[mask], color=col, alpha=0.3)

        ax.set_xlabel('Time, ns')
        ax.set_ylabel('Counts')
        ax.tick_params(axis='x', which='both', bottom=True, top=True)
        ax.tick_params(axis='y', which='both', left=True, right=True)



    '''custom colormaps so i dont have to recode it every time'''
    def gray_cmap(self):
        colors = [(1, 0, 0)] + [(i, i, i) for i in np.linspace(0, 1, 255)]
        return mcolors.LinearSegmentedColormap.from_list('custom_gray', colors, N=256)

    def seismic_cmap(self, center=None, range_val=2.5):
        colors = [(0, 0, 0)] + [plt.cm.seismic(i) for i in np.linspace(0, 1, 255)]
        cmap = mcolors.LinearSegmentedColormap.from_list('custom_seismic', colors, N=256)
        if center is not None:
            vmin = center - range_val
            vmax = center + range_val
            norm = mcolors.TwoSlopeNorm(vmin=vmin, vcenter=center, vmax=vmax)
            return cmap, norm
        return cmap
    
    def plasma_cmap(self, center=None, range_val=2.5):
        colors = [(0, 0, 0)] + [plt.cm.plasma(i) for i in np.linspace(0, 1, 255)]
        cmap = mcolors.LinearSegmentedColormap.from_list('custom_plasma', colors, N=256)
        if center is not None:
            vmin = center - range_val
            vmax = center + range_val
            norm = mcolors.TwoSlopeNorm(vmin=vmin, vcenter=center, vmax=vmax)
            return cmap, norm
        return cmap
    


    '''main plotter functions'''
    def plot_hist_unspliced(self, tau1, tau2, bins=20, filename='lifetime_histogram_spliced.png', show=True):
        tau1_flat = tau1.flatten()
        tau2_flat = tau2.flatten()

        # tau2_flat = tau2_flat[tau1_flat <= 2.5]
        # tau1_flat = tau1_flat[tau1_flat <= 2.5]  # remove values in tau1 less than 1
        # tau2_flat = tau2_flat[tau2_flat >= 1]

        print(f'tau1: {tau1_flat}')
        print(f'tau2: {tau2_flat}')

        # fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize=(10, 6), gridspec_kw={'width_ratios': [2, 2], 'wspace': 0.1})

        plt.hist(tau1_flat, bins=200, alpha=0.5, label=f'Tau1 (mean = {np.mean(tau1_flat):.2f} ns)', color='blue')
        plt.hist(tau2_flat, bins=bins, alpha=0.5, label=f'Tau2 (mean = {np.mean(tau2_flat):.2f} ns)', color='red')
        plt.xlabel('Lifetime (ns)')
        plt.xlim(0, 10)
        plt.ylabel('Frequency')

        plt.legend()
        plt.tight_layout()

        plt.savefig(filename)
        if show:
            plt.show()

    def plot_hist(self, tau1, tau2, bins=20, splice=(5, 20), filename='lifetime_histogram_spliced.png', show=True):
        tau1_flat = tau1.flatten()
        tau2_flat = tau2.flatten()

        # tau1_flat = tau1_flat[tau1_flat >= 1]  # remove values in tau1 less than 1
        # tau2_flat = tau2_flat[tau2_flat >= 1]

        left = splice[0]
        right = splice[1]

        fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize=(10, 6), gridspec_kw={'width_ratios': [2, 2], 'wspace': 0.1})

        ax1.hist(tau1_flat[tau1_flat < left], bins=bins, alpha=0.5, label=f'Tau1 (mean = {np.mean(tau1_flat):.2f} ns)', color='blue')
        ax1.hist(tau2_flat[tau2_flat < left], bins=bins, alpha=0.5, label=f'Tau2 (mean = {np.mean(tau2_flat):.2f} ns)', color='red')
        ax1.set_xlim(0, left)
        ax1.set_xlabel('Lifetime (ns)')
        ax1.set_ylabel('Frequency')
        ax1.set_title(f'Lifetimes < {left} ns')

        ax2.hist(tau1_flat[tau1_flat > right], bins=bins, alpha=0.5, color='blue')
        ax2.hist(tau2_flat[tau2_flat > right], bins=bins, alpha=0.5, color='red')
        ax2.set_xlim(right, np.max([np.max(tau1_flat), np.max(tau2_flat)]))
        ax2.set_xlabel('Lifetime (ns)')
        ax2.set_title(f'Lifetimes > {right} ns')

        ax1.spines['right'].set_visible(False)
        ax2.spines['left'].set_visible(False)
        ax1.tick_params(labelright=False)
        ax2.tick_params(labelleft=False)

        d = 0.02
        kwargs = dict(transform=ax1.transAxes, color='k', clip_on=False)
        ax1.plot((1 - d, 1 + d), (-d, +d), **kwargs)
        ax1.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)

        kwargs.update(transform=ax2.transAxes)
        ax2.plot((-d, +d), (-d, +d), **kwargs)
        ax2.plot((-d, +d), (1 - d, 1 + d), **kwargs)

        handles, labels = ax1.get_legend_handles_labels()
        fig.legend(handles, labels, loc='upper center')

        plt.tight_layout()

        plt.savefig(filename)
        if show:
            plt.show()

    def plot_mono(self, A1, tau1, intensity, full_trace, full_params, times, track, filename, show):
        fig, ax = plt.subplots(2, 2, figsize=(7, 7))
        fig.suptitle(f'{self.integ} us integ, {int(self.step)} ps step, {int(self.integ*self.numsteps*1e-3)} ms acq time, {self.thresh} thresh, {track} fits', fontsize=12)

        self.plot_image(ax[0, 0], A1, 'Amplitudes', 'cts', 'plasma')
        self.plot_image(ax[0, 1], intensity, 'Intensity', 'cts', self.gray_cmap(), mcolors.Normalize(vmin=0, vmax=np.max(intensity)))
        self.plot_image(ax[1, 0], tau1, 'Lifetimes', 'ns', self.seismic_cmap())
        self.plot_trace(ax[1, 1], times, full_trace, full_params, 'mono')

        for axi in ax.ravel():
            axi.set_xticks([])
            axi.set_yticks([])
        plt.tight_layout()
        plt.savefig(filename + '_results.png')
        if show:
            plt.show()

    def plot_bi(self, A, B, tau1, tau2, intensity, full_trace, full_params, times, track, filename, show):
        fig, ax = plt.subplots(2, 3, figsize=(11, 7))
        fig.suptitle(f'{self.integ} us integ, {int(self.step)} ps step, {int(self.integ*self.numsteps*1e-3)} ms acq time, {self.thresh} thresh, {track} fits', fontsize=12)

        self.plot_image(ax[0, 0], A, 'Amplitude', 'cts', self.plasma_cmap())
        self.plot_image(ax[0, 1], B, 'Weight', 'cts', self.plasma_cmap())
        self.plot_image(ax[1, 0], tau1, 'Larger Lifetime', 'ns', self.seismic_cmap(center=20, range_val=5))
        self.plot_image(ax[1, 1], tau2, 'Smaller Lifetime', 'ns', self.seismic_cmap(center=5, range_val=1.25))
        self.plot_image(ax[0, 2], intensity, 'Intensity', 'cts', self.gray_cmap(), mcolors.Normalize(vmin=0, vmax=np.max(intensity)))

        self.plot_trace(ax[1, 2], times, full_trace, full_params, 'bi')

        for i, axi in enumerate(ax.ravel()):
            print(i, axi)
            if i != 5: # remove ticks from all plots except trace
                axi.set_xticks([])
                axi.set_yticks([])


        plt.tight_layout()
        plt.savefig(filename + '_results.png')
        if show:
            plt.show()

    def plot_bi_rld(self, A1, A2, tau1, tau2, intensity, full_trace, full_params, times, track, filename, show):
        A1, A2, tau1, tau2 = self._swap_tau(A1, A2, tau1, tau2)

        fig, ax = plt.subplots(2, 3, figsize=(11, 7))
        fig.suptitle(f'{self.integ * 1e-3} ms integ, {int(self.step)} ns step, {int(self.integ*self.numsteps*1e-3)} ms acq time, {self.bits} bits, {(2 * self.kernel_size + 1)**2} pixels binned', fontsize=12)
        self.plot_image(ax[0, 0], A1, 'Smaller Amplitude', 'cts', self.plasma_cmap())
        self.plot_image(ax[0, 1], A2, 'Larger Amplitude', 'cts', self.plasma_cmap())
        self.plot_image(ax[1, 0], tau1, 'Smaller Lifetime', 'ns', self.seismic_cmap(center=5, range_val=5))
        self.plot_image(ax[1, 1], tau2, 'Larger Lifetime', 'ns', self.seismic_cmap(center=20, range_val=5))

        self.plot_image(ax[0, 2], intensity, 'Intensity', 'cts', self.gray_cmap(), mcolors.Normalize(vmin=0, vmax=np.max(intensity)))

        self.plot_rld(ax[1,2], times, full_trace, full_params, 'bi_rld')

        plt.tight_layout()
        plt.savefig(filename + '_results.png')
        if show:
            plt.show()



    '''plot coordinating function'''
    def plot_all(self, results, filename, show=False):
        A, B, tau1, tau2, intensity, full_trace, full_params, track, times = self.preprocess_results(results)
        if self.fit in ('mono', 'mono_conv', 'mono_conv_log', 'mono_conv_mh', 'mono_rld', 'mono_rld_50ovp'):
            self.plot_mono(A, tau2, intensity, full_trace, full_params, times, track, filename, show)
        elif self.fit in ('bi', 'bi_conv', 'bi_mh', 'bi_nnls', 'bi_nnls_conv'):
            self.plot_bi(A, B, tau1, tau2, intensity, full_trace, full_params, times, track, filename, show)
        elif self.fit in ('bi_rld'):
            self.plot_bi_rld(A, B, tau1, tau2, intensity, full_trace, full_params, times, track, filename, show)
