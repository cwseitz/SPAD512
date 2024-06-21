import numpy as np
import matplotlib.pyplot as plt
import subprocess
from scipy.io import savemat

class BNP:
    def __init__(self, config):
        self.config = config
        self.tau = None
        self.times = (np.arange(config['gate_num']) * config['gate_step']) + config['gate_offset']

    def image_BNP(self):
        image = np.zeros((5, 5))
        length, x, y = np.shape(image)

        for i in range(x):
            for j in range(y):
                trace = image[:self.config['gate_num'], i, j]

                if (np.sum(trace) > self.config['thresh']):
                    self.full_trace += trace
                    self.intensity[i][j] += np.sum(trace)

                    savemat('raw.mat', {'raw', trace})

                    cmd = [
                        "matlab", "-batch",
                        (
                            f"pixel_BNP('raw_data.mat', {self.config['PhCount']}, {self.config['Iter']}, "
                            f"{self.config['RatioThresh']}, {self.config['Number_species']}, {self.config['PI_alpha']}, "
                            f"{self.config['alpha_lambda']}, {self.config['beta_lambda']}, {self.config['freq']}, "
                            f"{self.config['irf_mean']}, {self.config['irf_sigma']}, {self.config['save_size']}, "
                            f"{self.config['gate_step']}, {self.config['gate_offset']})"
                        )
                    ]

                    result = subprocess.run(cmd, capture_output=True, text=True)

                    self.tau[i][j] += result.stdout
                    track += 1
                    print(result.stdout)
                    print(result.stderr)

        cmd = [
                "matlab", "-batch",
                (
                    f"pixel_BNP('raw_data.mat', {self.config['PhCount']}, {self.config['Iter']}, "
                    f"{self.config['RatioThresh']}, {self.config['Number_species']}, {self.config['PI_alpha']}, "
                    f"{self.config['alpha_lambda']}, {self.config['beta_lambda']}, {self.config['freq']}, "
                    f"{self.config['irf_mean']}, {self.config['irf_sigma']}, {self.config['save_size']}, "
                    f"{self.config['gate_step']}, {self.config['gate_offset']})"
                )
            ]

        result = subprocess.run(cmd, capture_output=True, text=True)

        full_tau = result.stdout
        print(result.stdout)
        print(result.stderr)

        return self.intensity, self.tau, full_tau, self.track

# testing code
config = {
    'gate_num': 10,
    'gate_step': 1,
    'gate_offset': 0,
    'thresh': 0.5,
    'PhCount': 5000,
    'Iter': 2500,
    'RatioThresh': 0.2,
    'Number_species': 5,
    'PI_alpha': 1,
    'alpha_lambda': 1,
    'beta_lambda': 50,
    'freq': 10,
    'irf_mean': 12,
    'irf_sigma': 12,
    'save_size': 5,
}

bnp = BNP(config)
intensity, tau, full_tau, track = bnp.image_BNP()

plt.imshow(tau, cmap='hot', interpolation='nearest')
plt.colorbar()
plt.title('Tau Image')
plt.show()