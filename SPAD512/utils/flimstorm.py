import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.optimize import curve_fit
from skimage.feature import peak_local_max
import matplotlib.pyplot as plt

class FLIMSTORM:
    def __init__(self, config: dict | None = None, **kwargs):
        cfg = {
            "freq": 40,
            "integ": 1000,
            "width": 5,
            "step": 5,
            "offset": 0,
            "numsteps": 10,
            "lifetimes": [1.5],
            "dark_cps": 200,
            "irf_mean": 0.0,
            "irf_width": 0.0,
            "fov_um": 10.0,
            "pixel_size_um": 0.1,
            "psf_fwhm_nm": 300.0,
            "n_emitters": 3,
            "zeta": 1.0,
            "seed": None,
        }
        if config:
            cfg.update(config)
        cfg.update(kwargs)
        for k, v in cfg.items():
            setattr(self, k, v)

        self.img_n = int(self.fov_um / self.pixel_size_um)
        self.x = self.y = self.img_n

        self.lifetimes = np.atleast_1d(self.lifetimes).astype(float)
        self.lambda_decay = 1.0 / (self.lifetimes[0] * 1e-9)

        self.gate_starts = (np.arange(self.numsteps) * self.step + self.offset) * 1e-9
        self.gate_duration_s = self.integ * 1e-6
        self.gate_width_s = self.width * 1e-9
        self.pulses_per_gate = int(self.freq * 1e6 * self.gate_duration_s)

        self.psf_sigma_px = (self.psf_fwhm_nm / 2.355) / (self.pixel_size_um * 1e3)
        self.rng = np.random.default_rng(self.seed)

    def _p_det(self, t0):
        lam = self.lambda_decay
        w = self.gate_width_s
        return np.exp(-lam * t0) * (1 - np.exp(-lam * w))

    def simulate_stack(self):
        stack = np.zeros((self.img_n, self.img_n, self.numsteps), dtype=np.uint16)
        emitters = self.rng.uniform(0, self.img_n, size=(self.n_emitters, 2))
        for x0, y0 in emitters:
            for g in range(self.numsteps):
                mean_ph = self.pulses_per_gate * self.zeta * self._p_det(self.gate_starts[g])
                n_phot = self.rng.poisson(mean_ph)
                if n_phot == 0:
                    continue
                xs = self.rng.normal(x0, self.psf_sigma_px, n_phot)
                ys = self.rng.normal(y0, self.psf_sigma_px, n_phot)
                xi = np.clip(np.round(xs).astype(int), 0, self.img_n - 1)
                yi = np.clip(np.round(ys).astype(int), 0, self.img_n - 1)
                np.add.at(stack[:, :, g], (yi, xi), 1)

        dark_lambda = self.dark_cps * self.gate_duration_s
        stack += self.rng.poisson(dark_lambda, size=stack.shape).astype(np.uint16)
        self.stack = stack
        self.truth_emitters_px = emitters
        return stack, emitters

    @staticmethod
    def _gauss2d(coords, amp, x0, y0, sigma, offset):
        x, y = coords
        return (offset + amp * np.exp(-((x-x0)**2 + (y-y0)**2) / (2*sigma**2))).ravel()

    def localise(self, expected=None):
        if expected is None:
            expected = self.n_emitters
        frame2d = self.stack.sum(axis=2) if hasattr(self, "stack") else None
        if frame2d is None:
            raise RuntimeError("Run simulate_stack() first.")
        sm = gaussian_filter(frame2d, sigma=1.0)
        peaks = peak_local_max(sm, min_distance=2, threshold_abs=0.2*sm.max())
        if len(peaks) > expected:
            intens = sm[tuple(peaks.T)]
            peaks = peaks[np.argsort(intens)[-expected:]]
        locs = []
        for y0, x0 in peaks:
            x_min, x_max = max(0, x0-3), min(self.img_n, x0+4)
            y_min, y_max = max(0, y0-3), min(self.img_n, y0+4)
            patch = frame2d[y_min:y_max, x_min:x_max]
            yy, xx = np.mgrid[y_min:y_max, x_min:x_max]
            guess = (patch.max()-patch.min(), x0, y0, self.psf_sigma_px, patch.min())
            try:
                popt, _ = curve_fit(self._gauss2d, (xx, yy), patch.ravel(), p0=guess,
                                    bounds=(0, np.inf), maxfev=2000)
                locs.append((popt[1], popt[2]))
            except RuntimeError:
                m = patch.sum(); locs.append(((xx*patch).sum()/m, (yy*patch).sum()/m))
        self.locs = np.array(locs)
        return self.locs

    def gate_counts(self, radius_px=2):
        yy, xx = np.mgrid[0:self.img_n, 0:self.img_n]
        curves = []
        for x_c, y_c in self.locs:
            mask = (xx - x_c)**2 + (yy - y_c)**2 <= radius_px**2
            curves.append(self.stack[mask].reshape(-1, self.numsteps).sum(axis=0))
        self.gate_curves = np.array(curves)
        return self.gate_curves

    def _fit_tau_single(self, curve):
        def model(t, lam, scale):
            return scale * np.exp(-lam*t) * (1 - np.exp(-lam*self.gate_width_s))
        popt, _ = curve_fit(model, self.gate_starts, curve,
                            p0=(self.lambda_decay, curve.max() or 1.0), bounds=(0, np.inf))
        return 1.0/popt[0], popt

    def fit_lifetimes(self):
        self.fit_results = [self._fit_tau_single(c) for c in self.gate_curves]
        self.tau_estimates_ns = [1e9 / fit[1][0] for fit in self.fit_results]
        return self.tau_estimates_ns

    def plot_results(self):
        frame2d = self.stack.sum(axis=2)
        plt.figure(figsize=(4,4))
        plt.imshow(gaussian_filter(frame2d, sigma=0.7), cmap="gray")
        if hasattr(self, "locs"):
            xs, ys = zip(*self.locs)
            plt.plot(xs, ys, 'x', color='red', markersize=6, label='localizations')
            plt.legend()
        plt.title("STORM")
        plt.colorbar(label="photons")
        plt.tight_layout()
        plt.show()

        plt.figure(figsize=(6,3))
        gate_times_ns = self.gate_starts * 1e9
        for idx, curve in enumerate(self.gate_curves):
            plt.plot(gate_times_ns, curve, 'o')
            lam, scale = self.fit_results[idx][1]
            fit_curve = scale * np.exp(-lam * self.gate_starts) * (1 - np.exp(-lam * self.gate_width_s))
            plt.plot(gate_times_ns, fit_curve, '-', color=plt.gca().lines[-1].get_color())
        plt.xlabel("time, ns")
        plt.ylabel("counts")
        plt.title("FLIM")
        plt.tight_layout()
        plt.show()

    def run(self, plot=True):
        self.simulate_stack()
        self.localise()
        self.gate_counts()
        self.fit_lifetimes()
        print("true coordinates:\n", np.round(self.truth_emitters_px, 2))
        print("localizations:\n", np.round(self.locs, 2))
        print("lifetimes:\n", np.round(self.tau_estimates_ns, 3))
        if plot:
            self.plot_results()

if __name__ == "__main__":
    cfg = {"freq": 40, "integ": 1000, "width": 5, "step": 0.2, "offset": 0,
           "numsteps": 10, "lifetimes": 1.5, "n_emitters": 20}
    sim = FLIMSTORM(cfg)
    sim.run()