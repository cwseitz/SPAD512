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

    def decay(self, x, amp, lam):
        return amp * np.exp(-x * lam)

    def decay_double(self, x, A, tau1, B, tau2):
        return A * (B * np.exp(-x / tau1) + (1 - B) * np.exp(-x / tau2))

    def preprocess_results(self, results):
        temp1 = results['A1'].astype(float)
        temp2 = results['A2'].astype(float)
        A1 = temp1*temp2
        A2 = temp1 - temp1*temp2

        tau1 = results['tau1'].astype(float)
        tau2 = results['tau2'].astype(float)
        intensity = results['intensity'].astype(float)
        full_trace = results['full_trace'].astype(float)
        full_params = results['full_params'].astype(float)
        track = results['track'].astype(int)
        times = results['times'].astype(float)

        full_params[1] = 1/(full_params[1] + 1e-10)
        full_params[3] = 1/(full_params[3] + 1e-10)

        k = self.kernel_size
        if k > 0:
            A1 = A1[k:-k, k:-k]
            A2 = A2[k:-k, k:-k]
            tau1 = tau1[k:-k, k:-k]
            tau2 = tau2[k:-k, k:-k]
            intensity = intensity[k:-k, k:-k]

        tau1, A1 = self._clamp_values(tau1, A1, 0, 100)
        tau2, A2 = self._clamp_values(tau2, A2, 0, 100)
        A1 = np.clip(A1, 0, 10000)
        A2 = np.clip(A2, 0, 10000)

        return A1, A2, tau1, tau2, intensity, full_trace, full_params, track, times

    def _clamp_values(self, tau, amp, min_val, max_val):
        mask = (tau > max_val) | (tau < min_val)
        tau[mask] = 0
        amp[mask] = 0
        return tau, amp

    def plot_mono(self, A1, tau1, intensity, full_trace, full_params, times, track, filename, show):
        fig, ax = plt.subplots(2, 2, figsize=(7, 7))
        fig.suptitle(f'{self.integ} us integ, {int(self.step)} ps step, {int(self.integ*self.numsteps*1e-3)} ms acq time, {self.thresh} thresh, {track} fits', fontsize=12)

        self._plot_image(ax[0, 0], A1, 'Amplitudes', 'cts', 'plasma')
        self._plot_image(ax[0, 1], intensity, 'Intensity', 'cts', self._custom_gray_colormap(), mcolors.Normalize(vmin=0, vmax=np.max(intensity)))
        self._plot_image(ax[1, 0], tau1, 'Lifetimes', 'ns', self._custom_seismic_colormap())
        self._plot_trace(ax[1, 1], times, full_trace, full_params, 'mono')

        for axi in ax.ravel():
            axi.set_xticks([])
            axi.set_yticks([])
        plt.tight_layout()
        plt.savefig(filename + '_results.png')
        if show:
            plt.show()

    def plot_bi(self, A1, A2, tau1, tau2, intensity, full_trace, full_params, times, track, filename, show):
        A1, A2, tau1, tau2 = self._swap_tau(A1, A2, tau1, tau2)
        fig, ax = plt.subplots(2, 3, figsize=(11, 7))
        fig.suptitle(f'{self.integ} us integ, {int(self.step)} ps step, {int(self.integ*self.numsteps*1e-3)} ms acq time, {self.thresh} thresh, {track} fits', fontsize=12)

        self._plot_image(ax[0, 0], A1, 'Smaller Amplitude', 'cts', self._custom_plasma_colormap())
        self._plot_image(ax[0, 1], A2, 'Larger Amplitude', 'cts', self._custom_plasma_colormap())
        self._plot_image(ax[1, 0], tau1, 'Smaller Lifetime', 'ns', self._custom_seismic_colormap(center=5, range_val=1.25))
        self._plot_image(ax[1, 1], tau2, 'Larger Lifetime', 'ns', self._custom_seismic_colormap(center=20, range_val=5))
        self._plot_image(ax[0, 2], intensity, 'Intensity', 'cts', self._custom_gray_colormap(), mcolors.Normalize(vmin=0, vmax=np.max(intensity)))

        self._plot_trace(ax[1, 2], times, full_trace, full_params, 'bi')

        for axi in ax.ravel():
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
        self._plot_image(ax[0, 0], A1, 'Smaller Amplitude', 'cts', self._custom_plasma_colormap())
        self._plot_image(ax[0, 1], A2, 'Larger Amplitude', 'cts', self._custom_plasma_colormap())
        self._plot_image(ax[1, 0], tau1, 'Smaller Lifetime', 'ns', self._custom_seismic_colormap(center=5, range_val=5))
        self._plot_image(ax[1, 1], tau2, 'Larger Lifetime', 'ns', self._custom_seismic_colormap(center=20, range_val=5))

        self._plot_image(ax[0, 2], intensity, 'Intensity', 'cts', self._custom_gray_colormap(), mcolors.Normalize(vmin=0, vmax=np.max(intensity)))

        self._plot_rld(ax[1,2], times, full_trace, full_params, 'bi_rld')

        plt.tight_layout()
        plt.savefig(filename + '_results.png')
        if show:
            plt.show()

    def _swap_tau(self, A1, A2, tau1, tau2):
        for i in range(len(A1)):
            for j in range(len(A1[0])):
                if tau2[i][j] < tau1[i][j]:
                    A1[i][j], A2[i][j] = A2[i][j], A1[i][j]
                    tau1[i][j], tau2[i][j] = tau2[i][j], tau1[i][j]
        return A1, A2, tau1, tau2

    def _plot_image(self, ax, data, title, cbar_label, cmap, norm=None):
        if isinstance(cmap, tuple):  # Check if cmap is a tuple containing (cmap, norm)
            cmap, norm = cmap
        im = ax.imshow(data, cmap=cmap, norm=norm)
        ax.set_title(title)
        plt.colorbar(im, ax=ax, label=cbar_label)

    def _plot_trace(self, ax, times, full_trace, full_params, fit_type):
        ax.set_title('Fully binned trace')
        
        full_trace /= np.max(full_trace)
        ax.scatter(times, full_trace, s=5)
        
        if fit_type == 'mono':
            A = full_params[0]
            lam = full_params[1]
            tau = 1 / lam
            ax.plot(times, self.decay(times, A, lam), label=f'Fit: tau = {tau:.2f} ns', color='black')
        
        elif fit_type == 'bi':
            A = full_params[0]
            lam1 = full_params[1]
            B = full_params[2]
            lam2 = full_params[3]
            tau1 = 1 / lam1
            tau2 = 1 / lam2
            ax.plot(times, self.decay_double(times, A, tau1, B, tau2), label=f'Fit: tau1 = {tau1:.2f} ns, tau2 = {tau2:.2f} ns', color='black')
        
        elif fit_type == 'mono_rld':
            A = full_params[0]
            lam = full_params[1]
            tau = 1 / lam
            ax.plot(times, self.decay(times, A, lam), label=f'Fit: tau = {tau:.2f} ns', color='black')
        
        # Add additional fit types as needed, matching the parameter structure returned by their fitting functions
        
        ax.set_xlabel('Time, ns')
        ax.set_ylabel('Counts')
        ax.set_ylim(0, 1.5 * np.max(full_trace))
        ax.tick_params(axis='x', which='both', bottom=True, top=True)
        ax.tick_params(axis='y', which='both', left=True, right=True)
        ax.legend()


    def _plot_rld(self, ax, times, full_trace, full_params, fit_type):
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

    def _custom_gray_colormap(self):
        colors = [(1, 0, 0)] + [(i, i, i) for i in np.linspace(0, 1, 255)]
        return mcolors.LinearSegmentedColormap.from_list('custom_gray', colors, N=256)

    def _custom_seismic_colormap(self, center=None, range_val=2.5):
        colors = [(0, 0, 0)] + [plt.cm.seismic(i) for i in np.linspace(0, 1, 255)]
        cmap = mcolors.LinearSegmentedColormap.from_list('custom_seismic', colors, N=256)
        if center is not None:
            vmin = center - range_val
            vmax = center + range_val
            norm = mcolors.TwoSlopeNorm(vmin=vmin, vcenter=center, vmax=vmax)
            return cmap, norm
        return cmap
    
    def _custom_plasma_colormap(self, center=None, range_val=2.5):
        colors = [(0, 0, 0)] + [plt.cm.plasma(i) for i in np.linspace(0, 1, 255)]
        cmap = mcolors.LinearSegmentedColormap.from_list('custom_plasma', colors, N=256)
        if center is not None:
            vmin = center - range_val
            vmax = center + range_val
            norm = mcolors.TwoSlopeNorm(vmin=vmin, vcenter=center, vmax=vmax)
            return cmap, norm
        return cmap
    
    def plot_hist(self, tau1, tau2, bins=50, splice=(5, 20), filename='lifetime_histogram_spliced.png', show=True):
        tau1_flat = tau1.flatten()
        tau2_flat = tau2.flatten()

        tau1_flat = tau1_flat[tau1_flat >= 1]  # Remove values in tau1 less than 1
        tau2_flat = tau2_flat[tau2_flat >= 1]

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


    def plot_all(self, results, filename, show=False):
        A1, A2, tau1, tau2, intensity, full_trace, full_params, track, times = self.preprocess_results(results)
        if self.fit in ('mono', 'mono_conv', 'mono_conv_log', 'mono_conv_mh', 'mono_rld', 'mono_rld_50ovp'):
            self.plot_mono(A2, tau2, intensity, full_trace, full_params, times, track, filename, show)
        elif self.fit in ('bi', 'bi_conv', 'bi_mh', 'bi_nnls', 'bi_nnls_conv'):
            self.plot_bi(A1, A2, tau1, tau2, intensity, full_trace, full_params, times, track, filename, show)
        elif self.fit in ('bi_rld'):
            self.plot_bi_rld(A1, A2, tau1, tau2, intensity, full_trace, full_params, times, track, filename, show)
