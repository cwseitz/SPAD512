import os
from PIL import Image
import tifffile as tiff
import numpy as np

base_dir = r"k:\\ishaan\\241101\\data\\gated_images"

subdirs = [
    d for d in os.listdir(base_dir)
    if os.path.isdir(os.path.join(base_dir, d)) and d.startswith('acq')
]

for subdir in subdirs:
    folder_path = os.path.join(base_dir, subdir)
    files = [f for f in os.listdir(folder_path) if f.endswith('.png')]
    files.sort()
    file_paths = [os.path.join(folder_path, f) for f in files]
    images = [np.asarray(Image.open(image)) for image in file_paths]
    output = os.path.join(base_dir, f"{subdir}_stacked.tif")
    tiff.imwrite(output, images)
    print(f"Saved TIFF movie as {output}")
