from PIL import Image
import tifffile as tiff
import numpy as np

files = [
    "C:\\Users\\ishaa\\Documents\\FLIM\\240825\\acq00003\\IMG00000-0000.png",
    "C:\\Users\\ishaa\\Documents\\FLIM\\240825\\acq00003\\IMG00000-0001.png",
    "C:\\Users\\ishaa\\Documents\\FLIM\\240825\\acq00003\\IMG00000-0002.png",
    "C:\\Users\\ishaa\\Documents\\FLIM\\240825\\acq00003\\IMG00000-0003.png"
]

images = [Image.open(image) for image in files]
images = [np.asarray(image) for image in images]

filename = "C:\\Users\\ishaa\\Documents\\FLIM\\240825\\240825_SPAD-QD-10MHz-1f-4g-50000us-15000ps-15000ps-18ps-150uW.tif"

tiff.imwrite(filename, images)

print(f"Saved TIFF movie as {filename}")
