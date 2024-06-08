from skimage.io import imread
import glob
import tifffile as tiff

folder = '240607/5_ms_6bit'
out = folder + '.tif'

with tiff.TiffWriter(out, bigtiff=True) as tif:
    for image in glob.glob(folder + '/*'):
        print(image)
        image = imread(image)
        tif.write(image, contiguous=True)

print('Image saved as ' + out)
