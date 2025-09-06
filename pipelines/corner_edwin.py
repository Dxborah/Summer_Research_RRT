import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

image = Image.open("../divided_world_1/15000-15999/dungeon_757.png").convert("L")
grid = np.array(image) // 255 

print(f"Image shape: {grid.shape}")
print(f"Unique values in grid: {np.unique(grid)}")

rows, cols = grid.shape
corner_positions = []

for i in range(rows - 1):
    for j in range(cols - 1):
        block = grid[i:i+2, j:j+2]
        if block.sum() == 3:
            black_idx = np.argwhere(block == 0)[0]
            di, dj = 1 - black_idx[0], 1 - black_idx[1]
            corner_i, corner_j = i + di, j + dj
            corner_positions.append((corner_i, corner_j))

plt.figure(figsize=(8, 8))
plt.imshow(1 - grid, cmap="gray", origin="upper") 
if corner_positions:
    ys, xs = zip(*corner_positions)
    plt.scatter(xs, ys, c='red', s=20, marker='o', label="270° corners")
plt.title("270° Corners of Obstacles")
plt.legend()
plt.axis("off")
plt.show()
