import numpy as np
from scipy.io import savemat
from skimage.io import imread, imsave
import matplotlib.pyplot as plt

def format(raw, start, bin_width, filename):
    data = []
    for i in range(len(raw)):
        count = raw[i]
        for _ in range(count):
            data.append(start + bin_width * i)

    np.random.shuffle(data)
    Dt = np.array(data).reshape(1, -1)
    savemat(filename, {'Dt': Dt})

    print(Dt)
    
    return Dt

image = imread("C://Users//shilp//OneDrive//Documents//240613_SPAD-QD-10MHz-3f-900g-10000us-5ns-100ps-18ps-150uW.tif")
image = image[:900,113,81]
print(image)

format(image, 0, 0.1, 'test.mat')