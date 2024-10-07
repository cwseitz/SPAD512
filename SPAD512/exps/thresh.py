import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from skimage.io import imread

filenames = "C:\\Users\\ishaa\\Documents\\FLIM\\240604\\240604_10ms_adjusted"
threshs = [10000]  # thresholds to test

# def lentrace(filename):
#     base_filename = filename.split('/')[-1]
#     base_filename = base_filename.split('.')[0]
#     parts = base_filename.split('-')

#     gate_num = int(parts[4].replace('g', ''))

#     return gate_num

def display_intensity_image(filename, thresh):
    image = imread(filename + '.tif')
    length, x, y = np.shape(image)
    # length = lentrace(filename + '.tif')
    length = 500
    image = image[:length, :, :]

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

if __name__ == '__main__':
    run_intensity_images(filenames, threshs)