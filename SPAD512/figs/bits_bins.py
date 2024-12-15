import numpy as np
import matplotlib.pyplot as plt

data_4bit = np.load("C:\\Users\\ishaa\\Documents\\FLIM\\241202\\4bit2_results.npz", allow_pickle=True)
data_6bit = np.load("C:\\Users\\ishaa\\Documents\\FLIM\\241202\\6bit_results.npz", allow_pickle=True)
data_8bit = np.load("C:\\Users\\ishaa\\Documents\\FLIM\\241202\\8bit_results.npz", allow_pickle=True)
data_8bit2 = np.load("C:\\Users\\ishaa\\Documents\\FLIM\\241202\\8bit2_results.npz", allow_pickle=True)

data = [data_4bit, data_6bit, data_8bit, data_8bit2]
integs = [0.1, 0.25, 1, 1]
colors = ['red', 'green', 'blue', 'blue']
labels = ['4bit', '6bit', '8bit', '8bit']
print(data[0]['metadata'])

for i, dt in enumerate(data):
    bins = dt['arr_bins']
    rsds = dt['tau1s']

    times = 10*bins*integs[i]

    plt.scatter(times, 1/rsds, color=colors[i], label=labels[i])

plt.xlabel('Acquisition time (ms)')
plt.ylabel('Inverse RSD of Smaller Lifetime')
plt.title('Number of bins versus relative precision')
plt.legend()
plt.grid(True)
plt.show()