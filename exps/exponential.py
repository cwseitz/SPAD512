import numpy as np
import scipy.optimize as opt
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from skimage.io import imsave, imread
from scipy.signal import deconvolve

'''
Script to fit deconvolved exponentials accross a widefield FLIM image and display the full results. 
To-do: 
- Automate threshold determination
- Choose better start point for exponential fitting
'''

# define parameters
filename = '240613_SPAD-QD-10MHz-3f-5000g-10ms-5ns-18ps-18ps-150uW' # no .tif
gate_num = 5000 # number of gate steps used
gate_step = 0.018 # gate step size in ns
thresh = 2500 # threshold for sum of counts over trace, below which there is no fitting done

freq = 10 # frequency in MHz
frames = 3 # number of frames
gate_integ = 100 # integration time in ns
gate_width = 5 # gate width in ns
gate_offset = 0.018 # gate offset in ps
irf_width = 0 # width of gaussian irf in ns


# exponential decay helper function
def decay(x, amp, tau):
  return amp * np.exp(-x / tau)

# scipy.curve_fit wrapper
def fit_decay(times, data):
  initial_guess = [np.max(data), 2.0] # use max for the amplitude, choice is ultimately insignificant with LMA algorithm
  params, cov = opt.curve_fit(decay, times, data, p0=initial_guess)
  return params

# deconvolution function
def irf_deconvolve(times, trace, irf_width):
  irf = np.exp(-times**2 / irf_width)
  tracedc = deconvolve(trace, irf)
  return tracedc

def fit_exps(filename, freq, frames, 
  gate_num, gate_integ, gate_width, 
  gate_step, gate_offset, irf_width, thresh):   
  # read image and set initial arrays
  image = imread(filename + '.tif')
  length, x, y = np.shape(image)

  times = (np.arange(gate_num) * gate_step) + gate_offset
  A = np.zeros((x, y))
  intensity = np.zeros((x, y))
  tau = np.zeros((x, y))
  full_trace = np.zeros((gate_num))
  track = 0

  # fit exponentials and save info
  for i in range(x):
    for j in range(y):
      trace = image[:gate_num, i, j]
      if (np.sum(trace) > thresh):
        full_trace += trace
        intensity[i][j] += sum(trace)

        loc = np.argmax(trace) # only fit from exponential peak onwards
        try: 
            params = fit_decay(times[loc:], trace[loc:])
            track += 1
        except RuntimeError:
            params = [0, 0] # not really sure how else to handle curve_fit failing after many iterations
        
        A[i][j] += params[0]
        tau[i][j] += params[1]
    
  loc = np.argmax(full_trace)
  try: 
    params = fit_decay(times[loc:], full_trace[loc:])
  except RuntimeError:
    params = [0, 0]

  print(str(track) + ' pixels successfully fit')
  return A, intensity, tau, times, full_trace, params

def plot_all(tau, A, intensity, gate_integ, gate_step, times, full_trace, params, filename):
  for i in range(len(tau)):
    for j in range(len(tau[0])):
      if tau[i][j] > 1000:
        tau[i][j] = 0
      if tau[i][j] < -1000:
        tau[i][j] = 0

  for i in range(len(A)):
    for j in range(len(A[0])):
      if A[i][j] > 100:
        A[i][j] = 0
      if A[i][j] < 0:
        A[i][j] = 0

  fig,ax=plt.subplots(2,2,figsize=(9,9))

  im1 = ax[0, 0].imshow(A,cmap='plasma')
  ax[0, 0].set_title('Amplitudes')
  plt.colorbar(im1,ax=ax[0, 0],label='cts')

  colors = [(1, 0, 0)] + [(i, i, i) for i in np.linspace(0, 1, 255)]
  custom = mcolors.LinearSegmentedColormap.from_list('custom_gray', colors, N=256)
  im2 = ax[0, 1].imshow(intensity,cmap=custom)
  ax[0, 1].set_title('Intensity')
  plt.colorbar(im2,ax=ax[0, 1],label='cts')

  ax[1, 0].set_title('Lifetimes')
  im3 = ax[1, 0].imshow(tau,cmap='hsv')
  plt.colorbar(im3, ax=ax[1, 0], label='ns')
  im3.set_clim(6, 20)

  ax[1, 1].set_title('Fully binned trace')
  ax[1, 1].scatter(times, full_trace, s=5)
  ax[1, 1].plot(times, decay(times, params[0], params[1]), label='Fit: tau = {:.2f}'.format(params[1]), color='black')
  ax[1, 1].set_xlabel('Time, ns')
  ax[1, 1].set_ylabel('Counts')
  ax[1, 1].tick_params(axis='x', which='both', bottom=True, top=True) # not working 
  ax[1, 1].tick_params(axis='y', which='both', left=True, right=True)
  ax[1, 1].legend()


  for axi in ax.ravel():
    axi.set_xticks([])
    axi.set_yticks([])

  plt.tight_layout()
  plt.savefig(filename + 'results.png')

  plt.show()
  
def run_exps(filename, freq, frames, gate_num, gate_integ, gate_width, gate_step, gate_offset, irf_width, thresh):
  A, intensity, tau, times, full_trace, params = fit_exps(filename, freq, frames, 
  gate_num, gate_integ, gate_width, 
  gate_step, gate_offset, irf_width, thresh)

  plot_all(tau, A, intensity, gate_integ, gate_step, times, full_trace, params, filename)

run_exps(filename, freq, frames, gate_num, gate_integ, gate_width, gate_step, gate_offset, irf_width, thresh)
