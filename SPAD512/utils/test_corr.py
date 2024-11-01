import numpy as np
import matplotlib.pyplot as plt
import tifffile as tf

filenames = [
            "k:\ishaan\\241101\data\gated_images\\acq00000_stacked.tif",
            # "k:\ishaan\\241101\data\gated_images\\acq00001_stacked.tif",
             "k:\ishaan\\241101\data\gated_images\\acq00002_stacked.tif",
            # "k:\ishaan\\241101\data\gated_images\\acq00003_stacked.tif",
             "k:\ishaan\\241101\data\gated_images\\acq00004_stacked.tif"
]
# filenames = [
#              "k:\ishaan\\241101\data\gated_images\\acq00015_stacked.tif"
# ]

integs = [
    25,
    75,
    10,
    5,
    1,
]

for i, filename in enumerate(filenames):
    print(f'this is i {i}')
    with tf.TiffFile(filename) as tif:
        image = tif.asarray()
        print(np.shape(image))  
    image = image[:,120:123,147:150]
    data=np.zeros(len(image))
    for j in range(len(image)):
        data[j] = np.mean(image[j,:,:])
    data2 = 63*(1 - np.exp(-data/63))
    plt.plot((0.5*np.arange(len(data)) + 0.018), data/np.max(data), label=f'post-correction for {integs[i]}')
    plt.plot((0.5*np.arange(len(data)) + 0.018), data2/np.max(data2), linestyle='--', label=f'inverted correction for {integs[i]}')
    # plt.plot(range(len(data)), np.log(data2/np.max(data2) - data/np.max(data)), label=f'{integs[i]}')

plt.legend()
plt.show()