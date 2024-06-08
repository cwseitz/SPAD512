from skimage.io import imread
import glob
import tifffile as tiff

folder = '240607/5_ms_6bit'
images = glob.glob(folder + '/*')

image = imread(image_files[0])
x, y = image.shape

with tiff.TiffWriter(folder + '.tif', bigtiff=True) as tif:
    for image in images:
        image = imread(image)
        tif.write(image, contiguous=True)

print('Image saved as ' + folder + '.tif')
