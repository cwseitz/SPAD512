# Short script to attach all images output by the SPAD into a single tiff, for piping into BNP or exponential analyses

import numpy as np 
from tifffile import imread, imsave, imwrite  
from pathlib import Path

files = Path('folder name here').glob('*')
image = imread(files[0])
for file in files:
    newimage = imread(file)
    image = np.hstack(image, file)
imwrite('image name here', image)
