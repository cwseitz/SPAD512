import numpy as np
import subprocess
from skimage.io import imread
import re
from multiprocessing import Pool, Manager
import json

class BNP:
    def __init__(self, config):
        self.config = config

    def process_pixel(self, trace, i, j, result_dict):
        results = {}
        if np.sum(trace) > self.config['thresh']:
            if not np.issubdtype(trace.dtype, np.number):
                trace = trace.astype(np.float64)
                
            trace_json = json.dumps(trace.tolist())

            cmd = [
                "matlab", "-batch",
                (
                    rf"addpath('C:\Users\ishaa\Documents\FLIM\SPAD512\SPAD512\bnps'); addpath('C:\Users\ishaa\Documents\FLIM\SPAD512\SPAD512\bnps\BNP-LA-main\Functions');"
                    f"pixel_BNP('{trace_json}', {self.config['PhCount']}, {self.config['Iter']}, "
                    f"{self.config['RatioThresh']}, {self.config['Number_species']}, {self.config['PI_alpha']}, "
                    f"{self.config['alpha_lambda']}, {self.config['beta_lambda']}, {self.config['freq']}, "
                    f"{self.config['irf_mean']}, {self.config['irf_sigma']}, {self.config['save_size']}, "
                    f"{self.config['gate_step']}, {self.config['gate_offset']})"
                )
            ]

            result = subprocess.run(cmd, capture_output=True, text=True)
            print(result.stdout)
            print(result.stderr)

            lifetime = re.search(r"Lifetime:\s*([\d\.]+)", result.stdout)
            if lifetime:
                val = float(lifetime.group(1))
                results['tau'] = (i, j, val)
                results['intensity'] = (i, j, np.sum(trace))
                results['track'] = 1
            else:
                print("Lifetime value not found in the output.")

        result_dict[(i, j)] = results

    def image_BNP(self):
        image = imread(self.config['filename'])
        length, x, y = np.shape(image)
        full_trace = np.zeros((length))
        self.full_tau = 0
        self.intensity = np.zeros((x, y))
        self.tau = np.zeros((x, y))
        self.track = 0

        manager = Manager()
        result_dict = manager.dict()

        num_workers = 2
        pool = Pool(num_workers)
        tasks = [
            pool.apply_async(self.process_pixel, (image[:self.config['gate_num'], i, j], i, j, result_dict))
            for i in range(x) for j in range(y)
        ]

        for task in tasks:
            task.wait()

        pool.close()
        pool.join()

        for key, result in result_dict.items():
            if 'tau' in result:
                i, j, val = result['tau']
                self.tau[i][j] += val
                self.track += result['track']
            if 'intensity' in result:
                i, j, intensity = result['intensity']
                self.intensity[i][j] += intensity

        return self.intensity, self.tau, self.track, self.full_tau

    def save_results(self):
        np.savez(f"{self.config['filename']}_bnp_results.npz", intensity=self.intensity, tau=self.tau, full_tau=self.full_tau, track=self.track)