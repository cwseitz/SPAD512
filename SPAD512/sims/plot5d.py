import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

np.random.seed(42)
n = 100

param1 = np.random.rand(n) * 10  # X-axis
param2 = np.random.rand(n) * 10  # Y-axis
param3 = np.random.rand(n) * 10  # Z-axis

param4 = np.random.rand(n) * 100

data_value = np.random.rand(n) * 1000
size = data_value / 50  # Scale the size for better visualization

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

scatter = ax.scatter(param1, param2, param3, c=param4, s=size, cmap='viridis', alpha=0.7)

cbar = plt.colorbar(scatter, ax=ax)
cbar.set_label('Integration')

ax.set_xlabel('Widths')
ax.set_ylabel('Offsets')
ax.set_zlabel('Steps')

plt.title('3D Scatter Plot with 4th Parameter as Color and Sphere Size for Data Value')

ax.view_init(elev=20., azim=-15, roll=0)

plt.show()
