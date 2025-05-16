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

def distance_angle(x1, y1, x2, y2):
    distance = math.sqrt(((x1 - x2)**2) + ((y1 - y2)**2))
    dy = y2 - y1
    dx = x2 - x1
    angle = math.atan(dy, dx)
    return (distance, angle)

def nearest_node(node_list, x, y):
    closest_node = None
    min_distance = float('inf') #setting minimum distance to large value

    for node in node_list:
        distance = math.sqrt(((node.x - x)**2) + ((node.y- y)**2))

        if distance < min_distance:
            min_distance = distance
            closest_node = node

    return closest_node

#moves toward target while limmitting step size
def steer(from_node, to_x, to_y, step_size):
    distance, angle = distance_angle(from_node.x, from_node.y, to_x, to_y)

    new_x = int(from_node.x + step_size * math.cos(angle))
    new_y = int(from_node.y + setp_size * math.sin(angle))

    return new_x, new_y