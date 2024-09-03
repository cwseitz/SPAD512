import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import itertools

np.random.seed(42)

steps = [1, 5]
offsets = [2, 6]
widths = [4, 12]
integs = [1, 10]

combinations = list(itertools.product(steps, offsets, widths, integs))

data = np.random.rand(len(combinations))

steps, offsets, widths, integs = zip(*combinations)

size = data * 100  

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

scatter = ax.scatter(steps, offsets, widths, c=integs, s=size, cmap='viridis', alpha=0.7)

cbar = plt.colorbar(scatter, ax=ax)
cbar.set_label('Integration')

ax.set_xlabel('Steps')
ax.set_ylabel('Offsets')
ax.set_zlabel('Widths')

plt.title('3D Scatter Plot with 4th Parameter as Color and Sphere Size for Data Value')

ax.view_init(elev=20., azim=-15)

plt.show()

# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# import itertools

# np.random.seed(42)

# # Define the lists
# steps = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
# offsets = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
# widths = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
# integs = [0.5, 1, 1.5, 2, 2.5, 5, 10, 25, 50, 100]

# # Generate all combinations of step, offset, width, and integ
# combinations = list(itertools.product(steps, offsets, widths, integs))

# # Generate random data for each combination
# data = np.random.rand(len(combinations))

# # Convert data into a structured grid
# data_grid = {}
# for (step, offset, width, integ), value in zip(combinations, data):
#     if integ not in data_grid:
#         data_grid[integ] = []
#     data_grid[integ].append([step, offset, value])

# fig, axs = plt.subplots(2, 5, figsize=(20, 10))
# axs = axs.flatten()

# for ax, integ in zip(axs, sorted(data_grid.keys())):
#     grid_data = np.array(data_grid[integ])
#     heatmap_data = grid_data[:, 2].reshape((len(steps), len(offsets)))  # Reshape to fit grid
#     sns.heatmap(heatmap_data, ax=ax, cmap='viridis', cbar=False, xticklabels=offsets, yticklabels=steps)
#     ax.set_title(f'Integration: {integ}')
#     ax.set_xlabel('Offsets')
#     ax.set_ylabel('Steps')

# fig.colorbar(axs[0].collections[0], ax=axs, location='right', aspect=40, label='Data Value')
# plt.tight_layout()
# plt.show()
