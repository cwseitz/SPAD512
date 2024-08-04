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
            'filename': 0,
        }
        defaults.update(config)
        defaults.update(kwargs)

        for key, val in defaults.items():
            setattr(self, key, val)

    def decay(self, x, amp, lam):
        return amp * np.exp(-x * lam)

    def decay_double(self, x, A, tau1, B, tau2):
        return A * (B * np.exp(-x / tau1) + (1 - B) * np.exp(-x / tau2))

    def preprocess_results(self, results):
        A1 = results['A1'].astype(float)
        A2 = results['A2'].astype(float)
        tau1 = results['tau1'].astype(float)
        tau2 = results['tau2'].astype(float)
        intensity = results['intensity'].astype(float)
        full_trace = results['full_trace'].astype(float)
        full_params = results['full_params'].astype(float)
        track = results['track'].astype(int)
        times = results['times'].astype(float)

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

        A1 = median_filter(A1, size=3)
        tau1 = median_filter(tau1, size=3)

        self._plot_image(ax[0, 0], A1, 'Amplitudes', 'cts', 'plasma')
        self._plot_image(ax[0, 1], intensity, 'Intensity', 'cts', self._custom_gray_colormap(), mcolors.Normalize(vmin=0, vmax=np.max(intensity)))
        self._plot_image(ax[1, 0], tau1, 'Lifetimes', 'ns', self._custom_seismic_colormap())

        self._plot_trace(ax[1, 1], times, full_trace, full_params)

        self._finplot(fig, ax, filename, show)

    def plot_bi(self, A1, A2, tau1, tau2, intensity, full_trace, full_params, times, track, filename, show):
        A1, A2, tau1, tau2 = self._swap_tau(A1, A2, tau1, tau2)

        fig, ax = plt.subplots(2, 3, figsize=(11, 7))
        fig.suptitle(f'{self.integ} us integ, {int(self.step)} ps step, {int(self.integ*self.numsteps*1e-3)} ms acq time, {self.thresh} thresh, {track} fits', fontsize=12)

        A1 = median_filter(A1, size=3)
        A2 = median_filter(A2, size=3)
        tau1 = median_filter(tau1, size=3)
        tau2 = median_filter(tau2, size=3)

        self._plot_image(ax[0, 0], A1, 'Smaller Amplitude', 'cts', self._custom_plasma_colormap())
        self._plot_image(ax[0, 1], A2, 'Larger Amplitude', 'cts', self._custom_plasma_colormap())
        self._plot_image(ax[1, 0], tau1, 'Smaller Lifetime', 'ns', self._custom_seismic_colormap())
        self._plot_image(ax[1, 1], tau2, 'Larger Lifetime', 'ns', self._custom_seismic_colormap())
        self._plot_image(ax[0, 2], intensity, 'Intensity', 'cts', self._custom_gray_colormap(), mcolors.Normalize(vmin=0, vmax=np.max(intensity)))

        self._plot_trace(ax[1, 2], times, full_trace, full_params)

        self._finplot(fig, ax, filename, show)

    def _swap_tau(self, A1, A2, tau1, tau2):
        for i in range(len(A1)):
            for j in range(len(A1[0])):
                if tau2[i][j] < tau1[i][j]:
                    A1[i][j], A2[i][j] = A2[i][j], A1[i][j]
                    tau1[i][j], tau2[i][j] = tau2[i][j], tau1[i][j]
        return A1, A2, tau1, tau2

    def _plot_image(self, ax, data, title, cbar_label, cmap, norm=None):
        im = ax.imshow(data, cmap=cmap, norm=norm)
        ax.set_title(title)
        plt.colorbar(im, ax=ax, label=cbar_label)

    def _plot_trace(self, ax, times, full_trace, full_params):
        ax.set_title('Fully binned trace')
        ax.scatter(times, full_trace, s=5)
        if self.fit in ('mono', 'mono_conv', 'mono_conv_log', 'mono_conv_mh'):
            ax.plot(times, self.decay(times, full_params[0], full_params[1]), label=f'Fit: tau = {1/full_params[1]:.2f}', color='black')
        elif self.fit in ('bi', 'bi_conv', 'bi_mh', 'bi_nnls', 'bi_nnls_conv', 'bi_rld'):
            ax.plot(times, self.decay_double(times, full_params[0], 1/full_params[1], full_params[2], 1/full_params[3]), label=f'Fit: tau = {1/full_params[1]:.2f}, {1/full_params[3]:.2f}', color='black')
        else:
            ax.plot(times, self.decay_double(times, full_params[0], 1/full_params[1], full_params[2], 1/full_params[3]), label=f'Fit: tau = {1/full_params[1]:.2f}, {1/full_params[3]:.2f}', color='black')

        ax.set_xlabel('Time, ns')
        ax.set_ylabel('Counts')
        ax.set_ylim(0, 1.5 * max(full_trace))
        ax.tick_params(axis='x', which='both', bottom=True, top=True)
        ax.tick_params(axis='y', which='both', left=True, right=True)
        ax.legend()

    def _finplot(self, fig, ax, filename, show):
        for i, axi in enumerate(ax.ravel()):
            if i != 3: # change to 5 somehow maybe without if logic but idk so many if statements here this is dumb
                axi.set_xticks([])
                axi.set_yticks([])
        plt.tight_layout()
        plt.savefig(filename + '_results.png')
        if show:
            plt.show()

    def _custom_gray_colormap(self):
        colors = [(1, 0, 0)] + [(i, i, i) for i in np.linspace(0, 1, 255)]
        return mcolors.LinearSegmentedColormap.from_list('custom_gray', colors, N=256)

    def _custom_seismic_colormap(self):
        colors = [(0, 0, 0)] + [plt.cm.seismic(i) for i in np.linspace(0, 1, 255)]
        return mcolors.LinearSegmentedColormap.from_list('custom_seismic', colors, N=256)

    def _custom_plasma_colormap(self):
        colors = [(0, 0, 0)] + [plt.cm.plasma(i) for i in np.linspace(0, 1, 255)]
        return mcolors.LinearSegmentedColormap.from_list('custom_plasma', colors, N=256)

    def plot_all(self, results, filename, show=False):
        A1, A2, tau1, tau2, intensity, full_trace, full_params, track, times = self.preprocess_results(results)
        if self.fit in ('mono', 'mono_conv', 'mono_conv_log', 'mono_conv_mh'):
            self.plot_mono(A1, tau1, intensity, full_trace, full_params, times, track, filename, show)
        elif self.fit in ('bi', 'bi_conv', 'bi_mh', 'bi_nnls', 'bi_nnls_conv', 'bi_rld'):
            self.plot_bi(A1, A2, tau1, tau2, intensity, full_trace, full_params, times, track, filename, show)
