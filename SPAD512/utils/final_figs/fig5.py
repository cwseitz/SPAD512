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

plt.figure(figsize=(10, 5))
for i, dt in enumerate(data):
    bins = dt['arr_bins']
    rsds = dt['tau1s']

    times = 10*bins*integs[i]/1000

    plt.scatter(times/1000, rsds, color=colors[i], label=labels[i])

plt.xlabel('Total Integration Time (s)', fontsize=14)
plt.xticks(ticks=[0, 0.2, 0.4, 0.6, 0.8, 1], labels=[0, 0.2, 0.4, 0.6, 0.8, 1], fontsize=12)
plt.ylabel('RSD of Smaller Lifetime', fontsize=14)
plt.yticks(ticks=[0.1, 0.2, 0.3, 0.4, 0.5], labels=[0.1, 0.2, 0.3, 0.4, 0.5], fontsize=12)
plt.legend(fontsize=16)
plt.show()