import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.optimize import curve_fit
from skimage.feature import peak_local_max
import matplotlib.pyplot as plt


class FLIMSTORM:
    def __init__(self, config=None, **kwargs):
        cfg = dict(
            freq=40,
            integ=1000,
            width=5,
            step=0.2,
            offset=0,
            numsteps=10,
            lifetimes=[1.5],
            dark_cps=200,
            irf_mean=0.0,
            irf_width=0.0,
            fov=10.0,
            pixel_size=0.1,
            psf_fwhm=300.0,
            n_emitters=50000,
            zeta=1.0,
            n_frames=2000,
            p_on_pulse=1e-9,
            p_off_pulse=4e-6,
            init_on_frac=0.00002
        )
        if config is not None:
            cfg.update(config)
        cfg.update(kwargs)
        for k, v in cfg.items():
            setattr(self, k, v)

        self.img_n = int(self.fov / self.pixel_size)
        self.x = self.y = self.img_n
        self.lifetimes = np.atleast_1d(self.lifetimes).astype(float)
        self.lambda_decay = 1.0 / (self.lifetimes[0] * 1e-9)

        self.gate_starts = (np.arange(self.numsteps) * self.step + self.offset) * 1e-9
        self.gate_duration_s = self.integ * 1e-6
        self.gate_width_s = self.width * 1e-9
        self.pulses_per_gate = int(self.freq * 1e6 * self.gate_duration_s)
        self.pulses_per_frame = self.pulses_per_gate * self.numsteps

        self.psf_sigma_px = (self.psf_fwhm / 2.355) / (self.pixel_size * 1e3)
        self.rng = np.random.default_rng(1)

        center = self.img_n / 2
        radius = self.img_n / 4
        angles = np.linspace(0, 2 * np.pi, self.n_emitters, endpoint=False)
        x_coords = center + radius * np.cos(angles)
        y_coords = center + radius * np.sin(angles)
        self.truth_emitters_px = np.stack((x_coords, y_coords), axis=1)

        self.state_on = self.rng.random(self.n_emitters) < self.init_on_frac

    def _p_detect(self, t0):
        lam = self.lambda_decay
        w = self.gate_width_s
        return np.exp(-lam * t0) * (1 - np.exp(-lam * w))

    def simulate_movie(self, show_each_frame=False):
        Nyx, Ng, Nf = self.img_n, self.numsteps, self.n_frames
        stack = np.zeros((Nyx, Nyx, Ng, Nf), dtype=np.uint16)

        p_det_gate = self._p_detect(self.gate_starts)
        photons_per_pulse = self.zeta * p_det_gate

        p_off_frame = 1 - (1.0 - self.p_off_pulse) ** self.pulses_per_frame
        p_on_frame = 1 - (1.0 - self.p_on_pulse) ** self.pulses_per_frame

        for f in range(Nf):
            print(f'generating frame {f}')
            turn_on = (~self.state_on) & (self.rng.random(self.n_emitters) < p_on_frame)
            turn_off = self.state_on & (self.rng.random(self.n_emitters) < p_off_frame)
            self.state_on[turn_on] = True
            self.state_on[turn_off] = False

            if not self.state_on.any():
                continue

            active_xy = self.truth_emitters_px[self.state_on]
            mean_photons_gate = self.pulses_per_gate * photons_per_pulse
            n_active = len(active_xy)

            for g in range(Ng):
                lam = mean_photons_gate[g]
                n_tot = self.rng.poisson(lam * n_active)
                if n_tot == 0:
                    continue

                idx = self.rng.integers(0, n_active, size=n_tot)
                emit_pos = active_xy[idx]
                offsets = self.rng.normal(0, self.psf_sigma_px, size=(n_tot, 2))
                blurred = emit_pos + offsets
                xi = np.clip(np.round(blurred[:, 0]).astype(np.int32), 0, Nyx - 1)
                yi = np.clip(np.round(blurred[:, 1]).astype(np.int32), 0, Nyx - 1)
                linear = yi * Nyx + xi
                hist = np.bincount(linear, minlength=Nyx * Nyx).reshape(Nyx, Nyx)
                stack[:, :, g, f] += hist.astype(np.uint16)

            dark_lambda = self.dark_cps * self.gate_duration_s
            stack[:, :, :, f] += self.rng.poisson(dark_lambda, size=(Nyx, Nyx, Ng)).astype(np.uint16)

            if show_each_frame:
                plt.figure(figsize=(3, 3))
                plt.imshow(stack[:, :, :, f].sum(axis=2), cmap="gray")
                plt.title(f"Frame {f}: {n_active} active")
                plt.axis("off")
                plt.tight_layout()
                plt.show()

        self.stack4d = stack
        return stack

    def _localise_frame(self, frame_idx, expected=100):
        frame2d = self.stack4d[:, :, :, frame_idx].sum(axis=2)
        sm = gaussian_filter(frame2d, sigma=1.0)
        peaks = peak_local_max(sm, min_distance=2, threshold_abs=0.2 * sm.max())
        if len(peaks) > expected:
            intens = sm[tuple(peaks.T)]
            peaks = peaks[np.argsort(intens)[-expected:]]
        locs = []
        for y0, x0 in peaks:
            x_min, x_max = max(0, x0 - 3), min(self.img_n, x0 + 4)
            y_min, y_max = max(0, y0 - 3), min(self.img_n, y0 + 4)
            patch = self.stack4d[y_min:y_max, x_min:x_max, :, frame_idx].sum(axis=2)
            yy, xx = np.mgrid[y_min:y_max, x_min:x_max]
            guess = (patch.max() - patch.min(), x0, y0, self.psf_sigma_px, patch.min())
            try:
                popt, _ = curve_fit(self.gauss2d, (xx, yy), patch.ravel(), p0=guess, bounds=(0, np.inf), maxfev=4000)
                locs.append((popt[1], popt[2]))
            except Exception:
                m = patch.sum()
                if m > 0:
                    locs.append(((xx * patch).sum() / m, (yy * patch).sum() / m))
        return np.array(locs)

    def localise_movie(self, expected_per_frame=100):
        self.all_localisations = []
        for f in range(self.n_frames):
            print(f'localizing frame {f}')
            locs = self._localise_frame(f, expected_per_frame)
            self.all_localisations.append(locs)
        return self.all_localisations

    @staticmethod
    def gauss2d(coords, amp, x0, y0, sigma, offset):
        x, y = coords
        return (offset + amp * np.exp(-((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))).ravel()

    def gate_counts(self, locs, radius_px=2, frame_idx=0):
        yy, xx = np.mgrid[0:self.img_n, 0:self.img_n]
        curves = []
        substack = self.stack4d[:, :, :, frame_idx]
        for x_c, y_c in locs:
            mask = (xx - x_c) ** 2 + (yy - y_c) ** 2 <= radius_px ** 2
            curves.append(substack[mask].reshape(-1, self.numsteps).sum(axis=0))
        return np.asarray(curves)

    def fit_tau_single(self, curve):
        def model(t, lam, scale):
            return scale * np.exp(-lam * t) * (1 - np.exp(-lam * self.gate_width_s))
        popt, _ = curve_fit(model, self.gate_starts, curve, p0=(self.lambda_decay, max(curve.max(), 1.0)), bounds=(0, np.inf))
        return 1.0 / popt[0], popt

    def run(self, show_each_frame=False, show_sr_plot=True, expected_per_frame=100, plot_overlay_frames=5):
        self.simulate_movie(show_each_frame=show_each_frame)
        self.localise_movie(expected_per_frame)

        self.all_curves = []
        self.all_lifetimes = []
        self.all_fits = []
        for f, locs in enumerate(self.all_localisations):
            if len(locs) == 0:
                continue
            curves = self.gate_counts(locs, frame_idx=f)
            self.all_curves.extend(curves)
            fits = [self.fit_tau_single(c) for c in curves]
            self.all_fits.extend(fits)
            self.all_lifetimes.extend([fit[0] for fit in fits])

        self.all_localised_points = np.vstack([locs for locs in self.all_localisations if len(locs)])
        self.all_lifetimes_ns = np.array(self.all_lifetimes) * 1e9

        if show_sr_plot and len(self.all_localised_points):
            plt.figure(figsize=(5, 5))
            sc = plt.scatter(self.all_localised_points[:, 0], self.all_localised_points[:, 1],
                            c=self.all_lifetimes_ns, cmap="turbo", s=4, alpha=0.8)
            ax = plt.gca()
            ax.set_facecolor("black")  # set interior to black
            plt.colorbar(sc, label="lifetime (ns)")
            plt.title("All localizations colored by τ", color="white")
            plt.axis("equal")
            plt.tight_layout()
            plt.show()

        for f in range(min(plot_overlay_frames, self.n_frames)):
            locs = self.all_localisations[f]
            frame2d = self.stack4d[:, :, :, f].sum(axis=2)
            plt.figure(figsize=(4, 4))
            plt.imshow(gaussian_filter(frame2d, sigma=0.7), cmap="gray")
            if len(locs):
                xs, ys = zip(*locs)
                plt.plot(xs, ys, 'x', color='cyan', markersize=5, label='localizations')
                plt.legend()
            plt.title(f"Frame {f}")
            plt.axis("equal")
            plt.tight_layout()
            plt.show()

        # for idx, (curve, (_, popt)) in enumerate(zip(self.all_curves, self.all_fits)):
        #     t_ns = self.gate_starts * 1e9
        #     fit_curve = popt[1] * np.exp(-popt[0] * self.gate_starts) * (1 - np.exp(-popt[0] * self.gate_width_s))
        #     plt.figure(figsize=(5, 3))
        #     plt.plot(t_ns, curve, 'ok')
        #     plt.plot(t_ns, fit_curve)
        #     plt.xlabel("Time (ns)")
        #     plt.ylabel("Counts")
        #     plt.title(f"Gate trace with fit")
        #     plt.legend()
        #     plt.tight_layout()
        # plt.show()

    def not_storm(self, blur_sigma_px=1.5, brightness=1.0):
        img = np.zeros((self.img_n, self.img_n), dtype=np.float32)
        for x, y in self.truth_emitters_px:
            xi = int(round(x))
            yi = int(round(y))
            if 0 <= xi < self.img_n and 0 <= yi < self.img_n:
                img[yi, xi] += brightness

        sigma = blur_sigma_px if blur_sigma_px is not None else self.psf_sigma_px
        img_blurred = gaussian_filter(img, sigma=sigma)

        plt.figure(figsize=(5, 5))
        plt.imshow(img_blurred, cmap="gray", origin="lower")
        plt.colorbar()
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    cfg = dict(
        freq=40, integ=1000, width=5, step=0.2, offset=0,
        numsteps=10, n_emitters=50000, n_frames=10,
        p_on_pulse=1e-9, p_off_pulse=4e-6, init_on_frac=0.00002,
        lifetimes=1.5,
    )
    sim = FLIMSTORM(cfg)
    sim.run(show_each_frame=False, show_sr_plot=True, plot_overlay_frames=4)
    sim.not_storm()