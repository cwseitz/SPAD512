import numpy as np
from skimage.io import imread, imsave 

def rebin(raw, out, ratio=1000):
  image = imread(raw)
  length, x, y = image.shape
  newlen = length // ratio

  new = np.zeros((newlen, x, y), dtype=image.dtype)

  for i in range(newlen):
    new[i] = np.sum(image[i*ratio:(i+1)*ratio], axis=0)
    print(f'set {i} done')
  print(f'New shape: {np.shape(new)}')

  imsave(out, new)
  print(f'Rebinned image saved as {out}')
