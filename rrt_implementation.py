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

def collision(x1, y1, x2, y2, grid):
    #generates 100 evenly spaced points for each axis
    x_points = np.linspace(x1, x2, 100).astype(int)
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
    angle = math.atan2(dy, dx)
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
    new_y = int(from_node.y + step_size * math.sin(angle))

    return new_x, new_y

#main function
def rrt(grid, start, goal, step_size=5, max_iter=500):
    start_node = Node(*start)
    goal_node = Node(*goal)
    nodes = [start_node]

    for _ in range(max_iter):
        rand_x, rand_y = random.randint(0, grid.shape[1]-1), random.randint(0, grid.shape[0]-1)
        nearest = nearest_node(nodes, rand_x, rand_y)
        new_x, new_y = steer(nearest, rand_x, rand_y, step_size)

        #checking if new node collides w/ obstacles
        if not collision(nearest.x, nearest.y, new_x, new_y, grid):
            new_node = Node(new_x, new_y)
            new_node.parent = nearest
            nodes.append(new_node)

            #check if new node can directly connect to the goal
            if not collision(new_x, new_y, goal_node.x, goal_node.y, grid):
                goal_node.parent = new_node
                nodes.append(goal_node)
                print("Path found!")
                return nodes
            
    print("Path not found")
    return None

#Extract path
def path(goal_node):
    path = []
    node = goal_node
    while node is not None:
        path.append((node.x, node.y))
        node = node.parent

    path.reverse() #start->goal
    return path

#Running main function RRT
start= (10, 10)
goal = (90, 90)
nodes = rrt(grid, start, goal, step_size=5)

#visualize
if nodes:
    path = path(nodes[-1])
    
    explored_x = [node.x for node in nodes]
    explored_y = [node.y for node in nodes]

    path_x = [x for x, y in path]
    path_y = [y for x, y in path]

    plt.imshow(grid, cmap='gray')
    plt.scatter(explored_x, explored_y, color='red')
    plt.plot(path_x, path_y, color='blue', linewidth=2)
    plt.title("RRT")

    #plt.savefig("rrt_path2.png", dpi=300, bbox_inches='tight')
    plt.show(block=True)

    #input("Press Enter to exit")