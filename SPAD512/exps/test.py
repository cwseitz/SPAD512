import numpy as np

x = np.zeros((10, 10))

ksize = 1

track = 0
for i in range(len(x)):
    for j in range(len(x[0])):
        track += 1
        x[i][j] += track

print(x)
for i in range(ksize, len(x)-ksize):
    for j in range(ksize, len(x[0])-ksize):
        print(x[(i-ksize):(i+ksize+1), (j-ksize):(j+ksize+1)])