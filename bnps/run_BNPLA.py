import numpy as np # type: ignore
import scipy.io as sio # type: ignore
from skimage.io import imread #type: ignore
import random
import matlab.engine
import numpy as np
import scipy.io as sio

x = 147 # x coordinate of interest
y = 215 # y coordinate of interest
bin_width = 0.09 # spacing of bins in nanoseconds (NOT SPAD512 gate width)
start = 0 # start point for bins (taken as half of gate width plus initial offset)
file = '240604/240604_10ms_adjusted.tif'

image = imread(file)
length, xdim, ydim = image.shape
raw = image[:, x, y]
data = []
for i, count in enumerate(raw):
    for j in range(int(count)):
        data.append(start + bin_width * i)

random.shuffle(data) # needed so that when not using all photons, data isn't skewed to taking early times only

data = np.array(data).reshape(1, -1)
print(data)

# import matlab.engine
# import numpy as np
# import scipy.io as sio

# def run_matlab_gui_with_data(pixel_data, num_iterations, num_species):
#     eng = matlab.engine.start_matlab()
#     guiFig, Data = eng.gui(nargout=2)

#     eng.set_param(Data, 'T_max', 12.8)
#     eng.set_param(Data, 't_p', 12.2)
#     eng.set_param(Data, 'sigma_p', 0.66)
#     eng.set_param(Data, 'Save_size', num_species)
#     eng.set_param(Data, 'Number_species', num_species)
#     eng.set_param(Data, 'alpha_lambda', 1)
#     eng.set_param(Data, 'beta_lambda', 50)
#     eng.set_param(Data, 'Prop_lambda', 1000)
#     eng.set_param(Data, 'Ntmp', 5)

#     pixel_data_matlab = matlab.double(pixel_data.tolist()) # maybe unneeded
#     eng.set(Data, 'DtAll', pixel_data_matlab)
#     eng.set(Data, 'Iter', num_iterations)
#     eng.eval("runCode()", nargout=0)
#     results = eng.workspace['Data']
#     eng.quit()
    
#     return results

# # Example usage
# # Generate some pixel data
# pixel_data = np.random.rand(1000)  # Replace with actual pixel data
# num_iterations = 2500
# num_species = 5

# results = run_matlab_gui_with_data(pixel_data, num_iterations, num_species)

# # Save results if needed
# sio.savemat('processed_results.mat', {'results': results})
