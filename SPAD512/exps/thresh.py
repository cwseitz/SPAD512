import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from skimage.io import imread

# Define parameters
filenames = 'mains\\240613_SPAD-QD-10MHz-3f-900g-1000us-5ns-100ps-18ps-150uW'
threshs = [100, 200, 300, 400, 500]  # Thresholds for sum of counts over trace

def lentrace(filename):
    # Split filename into individual values
    base_filename = filename.split('/')[-1]
    base_filename = base_filename.split('.')[0]
    parts = base_filename.split('-')

    gate_num = int(parts[4].replace('g', ''))

    return gate_num

def display_intensity_image(filename, thresh):
    # Read image and set initial arrays
    image = imread(filename + '.tif')
    length, x, y = np.shape(image)
    length = lentrace(filename + '.tif')

    intensity = np.zeros((x, y))

    for i in range(x):
        for j in range(y):
            trace = image[:length, i, j]
            if np.sum(trace) > thresh:
                intensity[i][j] = np.sum(trace)

    plt.figure(figsize=(7, 7))
    colors = [(1, 0, 0)] + [(i, i, i) for i in np.linspace(0, 1, 255)]
    custom = mcolors.LinearSegmentedColormap.from_list('custom_gray', colors, N=256)
    plt.imshow(intensity, cmap=custom)
    plt.title(f'Intensity Image for Threshold: {thresh}')
    plt.colorbar(label='cts')
    plt.xticks([])
    plt.yticks([])
    plt.show()

def run_intensity_images(filename, threshs):
    for thresh in threshs:
        display_intensity_image(filename, thresh)

run_intensity_images(filenames, threshs)
