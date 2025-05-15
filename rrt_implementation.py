import numpy as np
import matplotlib.pyplot as plt
import math
import random

class Node:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.parent = None
        

grid_size = 100 #change this to liking
grid = np.ones((grid_size, grid_size), dtype=np.uint8) * 255  

#adding black obstacles
grid[20:40, 20:40] = 0
grid[60:80, 10:30] = 0
grid[50:70, 60:90] = 0

plt.imshow(grid, cmap='gray')
plt.title("RRT")
plt.show()

def collision(x1, y1, x2, y2, grid):
    #generates 100 evenly spaced points for each axis
    x_points = np.linespace(x1, x2, 100).astype(int)
    y_points = np.linspace(y1, y2, 100).astype(int)

    #iterates over each x,y pair
    for x, y in zip(x_points, y_points):
        if x < 0 or x >=  grid.shape[1] or y < 0 or y >=  grid.shape[0]:
            return True #out of bounds
        if grid[y, x] == 0:
            return True  #obstacle
    return False

def 