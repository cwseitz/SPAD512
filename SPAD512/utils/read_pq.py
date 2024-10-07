import ptufile
import matplotlib.pyplot as plt
import numpy as np

filename = 'c:\\Users\\ishaa\\Documents\\FLIM\\picoquant\\LSM_2.ptu'
ptu = ptufile.PtuFile(filename)

x = ptu.decode_histogram(dtype='uint8')

histogram_data = x.flatten()

plt.figure(figsize=(10, 6))
plt.plot(histogram_data, label='Histogram Data')
plt.xlabel('Bin Index')
plt.ylabel('Counts')
plt.title('Histogram of PTU Data')
plt.legend()
plt.grid(True)
plt.show()
