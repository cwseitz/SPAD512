import numpy as np
import tifffile as tf
from concurrent.futures import ProcessPoolExecutor, as_completed
import time
from SPAD512.exps import Trace
    
class Fitter:
    def __init__(self,config,**kwargs):
        defaults = {
            'step': 0,
            'fit': "",
            'irf_width': 0,
            'irf_mean': 0,
            'thresh': 0,
            'width': 0,
            'kernel_size': 0,
            'numsteps': 0,
            'times': 0,
            'offset': 0
        }
        defaults.update(config)
        defaults.update(kwargs)

        for key, val in defaults.items():
            setattr(self,key,val)

        self.step *= 1e-3 # ps --> ns
        self.width *= 1e-3 
        self.offset *= 1e-3

        print(self.__dict__)

        self.A1 = None
        self.tau1 = None
        self.A2 = None
        self.tau2 = None
        self.intensity = None
        self.full_trace = None
        self.track = 0    

    '''Parallelizing helper function'''
    def helper(self, data, i, j):
        length, x, y = np.shape(data)
        
        data_knl = np.zeros(length)
        for a in range(x):
            for b in range(y):
                data_knl += data[:, a, b]

        dt = Trace(data_knl, i, j, **self.__dict__)
        dt.fit_trace()
        return dt.params, dt.success, dt.sum, dt.i, dt.j

    def fit_exps(self, filename=None, image=None):
        tic = time.time()
        print('Reading image')
        if filename:
            with tf.TiffFile(filename + '.tif') as tif:
                image = tif.asarray(key=range(self.numsteps))  # Only read the first 5000 frames
            length, x, y = np.shape(image)
        elif image is not None:
            image = image[:self.numsteps,:,:]
            length, x, y = np.shape(image)
        else:
            raise Exception('No filename or image provided to fit_exps, make sure to provide one or the other.')
        toc = time.time()
        print(f'Image read in {(toc-tic):.1f} seconds')

        self.A1 = np.zeros((x, y), dtype=float)
        self.A2 = np.zeros((x, y), dtype=float)
        self.tau1 = np.zeros((x, y), dtype=float)
        self.tau2 = np.zeros((x, y), dtype=float)
        self.intensity = np.zeros((x, y), dtype=float)
        self.full_trace = np.zeros((self.numsteps), dtype=float)

        with ProcessPoolExecutor() as executor:
            futures = [executor.submit(self.helper, image[:, (i-self.kernel_size):(i+self.kernel_size+1), (j-self.kernel_size):(j+self.kernel_size+1)], i, j) for i in range(self.kernel_size,x-self.kernel_size) for j in range(self.kernel_size, y-self.kernel_size)]

            for future in as_completed(futures):
                outputs, success, sum, i, j = future.result()
                if success:
                    self.A1[i][j] += outputs[0]
                    self.tau1[i][j] += 1/(outputs[1]+1e-10)
                    self.A2[i][j] += outputs[2]
                    self.tau2[i][j] += 1/(outputs[3]+1e-10)
                    self.intensity[i][j] += sum

                    self.full_trace += image[:, i, j]
                    self.track += 1
                    print(f'Pixel ({i}, {j}): {1/(outputs[1]+1e-10)} ns\n')

        full_reshaped = self.full_trace.reshape(len(self.full_trace),1,1)

        outputs, success, sum, i, j = self.helper(full_reshaped, 0, 0)

        return self.A1, self.A2, self.tau1, self.tau2, self.intensity, self.full_trace, outputs, self.track, self.times
    
    def save_results(self, filename, results):
        np.savez(
            filename + '_fit_results.npz', 
            A1=results[0], 
            A2=results[1], 
            tau1=results[2], 
            tau2=results[3], 
            intensity=results[4], 
            full_trace=results[5], 
            full_params=results[6], 
            track=results[7],
            times=results[8]
        )   