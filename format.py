import numpy as np # type: ignore
from skimage.io import imread, imsave #type: ignore

raw = '240607/5_ms_6bit.tif'
out = '240607/5_ms_6bit_processed.tif'
ratio = 5

image = imread(raw)
length, x, y = image.shape
newlen = length // ratio

new = np.zeros((newlen, x, y), dtype=image.dtype)

for i in range(newlen):
  new[i] = np.sum(image[i*ratio:(i+1)*ratio], axis=0)
  print(f'set {i} done')
print(f'New shape: {np.shape(new)}')

imsave(out, new)
print(f'Image saved as {out}')