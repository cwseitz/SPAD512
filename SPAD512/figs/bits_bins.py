import numpy as np
import matplotlib.pyplot as plt

data_4bit = np.load("C:\\Users\\ishaa\\Documents\\FLIM\\241202\\4bit2_results.npz", allow_pickle=True)
data_6bit = np.load("C:\\Users\\ishaa\\Documents\\FLIM\\241202\\6bit_results.npz", allow_pickle=True)
data_6bit2 = np.load('c:\\Users\\ishaa\\Documents\\FLIM\\241202\\6bit_part2.npz', allow_pickle=True)
data_8bit = np.load("C:\\Users\\ishaa\\Documents\\FLIM\\241202\\8bit_results.npz", allow_pickle=True)
data_8bit2 = np.load("C:\\Users\\ishaa\\Documents\\FLIM\\241202\\8bit2_results.npz", allow_pickle=True)
data_12bit = np.load("C:\\Users\\ishaa\\Documents\\FLIM\\241202\\12bit_results.npz", allow_pickle=True)

data = [data_4bit, data_6bit, data_6bit2, data_8bit, data_8bit2, data_12bit]
integs = [dt['metadata'].item()['integ'] for dt in data]
colors = ['red', 'green', 'green', 'blue', 'blue', 'orange']
labels = ['4bit', '6bit', None, '8bit', None, '12bit']
print(data[5]['arr_bins'])

for i, dt in enumerate(data):
    bins = dt['arr_bins']
    rsds = dt['tau1s']

    times = 10*bins*integs[i]/1000

    plt.scatter(times, rsds, color=colors[i], label=labels[i])

plt.xlabel('Total integration time (ms)')
plt.ylabel('RSD of Smaller Lifetime')
plt.title('Integration time versus relative precision')
plt.legend()
plt.grid(True)
plt.show()