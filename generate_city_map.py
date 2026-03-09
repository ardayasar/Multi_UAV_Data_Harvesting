import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# Load the RBM map data
buildings, map_x_len, map_y_len, grid_x, grid_y, height_map = np.load(
    '/Users/alparslanguzey/Desktop/Multi_UAV_Data_Harvesting/config/RBM_map.npy', allow_pickle=True
)

# Determine grid size from meshgrid spacing
grid_size = grid_x[0, 1] - grid_x[0, 0]

# Create a category grid: 0=street, 1=building
cat = np.zeros_like(height_map, dtype=int)
cat[height_map > 0] = 1

# Sample positions in meters
uav_start = [280, 300]
UAV_start_idx = (int(uav_start[1]/grid_size), int(uav_start[0]/grid_size))
cat[UAV_start_idx] = 2  # UAV start

uav_terminal = [320, 460]
UAV_term_idx = (int(uav_terminal[1]/grid_size), int(uav_terminal[0]/grid_size))
cat[UAV_term_idx] = 3  # UAV terminal

anchor_devices = [[100, 300], [200, 600], [300, 800]]
for x, y in anchor_devices:
    i, j = int(y/grid_size), int(x/grid_size)
    cat[i, j] = 4

unknown_devices = [[50, 100], [150, 400], [250, 500]]
for x, y in unknown_devices:
    i, j = int(y/grid_size), int(x/grid_size)
    cat[i, j] = 5

# Define discrete colormap
cmap = ListedColormap([
    'white',      # 0: Street
    'lightgray',  # 1: Building
    'lightblue',  # 2: UAV Start
    'cyan',       # 3: UAV Terminal
    'orange',     # 4: Anchor Device
    'brown'       # 5: Unknown Device
])

# Plot grid map
fig, ax = plt.subplots(figsize=(6, 8))
ax.imshow(
    cat.T, origin='lower', extent=(0, map_x_len, 0, map_y_len), cmap=cmap, interpolation='none'
)

# Draw grid lines
nx, ny = cat.shape
ax.set_xticks(np.arange(0, map_x_len + grid_size, grid_size))
ax.set_yticks(np.arange(0, map_y_len + grid_size, grid_size))
ax.grid(which='major', color='k', linewidth=0.5)

ax.set_xlabel('X-axis [m]')
ax.set_ylabel('Y-axis [m]')
plt.tight_layout()

# Save output
output_path = '/mnt/data/RBM_Grid_Q1.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight')
plt.show()
print(f"Saved grid map to {output_path}")
