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

def random_point(grid_size, goal, goal_sample_rate=0.05):
    if random.random() < goal_sample_rate:
        return goal  # bias towards goal
    return (random.randint(0, grid_size-1), random.randint(0, grid_size-1))


#moves toward target while limmitting step size
def steer(from_node, to_x, to_y, step_size):
    distance, angle = distance_angle(from_node.x, from_node.y, to_x, to_y)
    distance = min(distance, step_size)
    new_x = int(from_node.x + distance * math.cos(angle))
    new_y = int(from_node.y + distance * math.sin(angle))

    # keep on grid limits
    new_x = max(0, min(new_x, grid.shape[1] - 1))
    new_y = max(0, min(new_y, grid.shape[0] - 1))

    return new_x, new_y

#main function
def rrt(grid, start, goal, step_size=10, max_iter=1000):
    start_node = Node(start[0], start[1])
    goal_node = Node(goal[0], goal[1])
    nodes = [start_node]

    for _ in range(max_iter):
        rand_x, rand_y = random_point(grid.shape[0], goal)
        nearest = nearest_node(nodes, rand_x, rand_y)
        new_x, new_y = steer(nearest, rand_x, rand_y, step_size)

        #checking if new node collides w/ obstacles
        if not collision(nearest.x, nearest.y, new_x, new_y, grid):
            new_node = Node(new_x, new_y)
            new_node.parent = nearest
            nodes.append(new_node)

            #check if new node can directly connect to the goal
            if math.hypot(new_x - goal[0], new_y - goal[1]) < step_size:
                goal_node.parent = new_node
                nodes.append(goal_node)
                print("Path found!")
                return nodes
            
    print("Path not found")
    return None

def interpolate_path(x1, y1, x2, y2):
    """
    Interpolates a path along grid lines using 1x1 Manahttan steps
    d = |x_1 - x_2| + |y_1 - y_2| -> formula
    """
    path = []
    x, y = x1, y1
    dx = np.sign(x2-x1)
    dy = np.sign(y2 - y1)

    while x!= x2:
        x += dx
        path.append((x, y))

    while y != y2:
        y += dy
        path.append((x, y))

    return path

def collision(x1, y1, x2, y2, grid):
    path = interpolate_path(x1, y1, x2, y2)
    #iterates over each x,y pair
    for x, y in path:
        if x < 0 or x >=  grid.shape[1] or y < 0 or y >=  grid.shape[0]:
            return True #out of bounds
        if grid[y, x] == 0:
            return True  #obstacle
    return False

def draw_tree(nodes, ax):
    for node in nodes:
        if node.parent is not None:
            steps = interpolate_path(node.parent.x, node.parent.y, node.x, node.y)
            step_x, step_y = zip(*steps)
            ax.plot(step_x, step_y, color='orange', linewidth=0.5)
            ax.scatter(step_x, step_y, color='orange', s=1)  # dots for every step taken

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
    final_path = path(nodes[-1])

    fig, ax = plt.subplots()
    ax.imshow(grid, cmap='gray')

    # Draw RRT tree
    draw_tree(nodes, ax)

    # Scatter all explored nodes (tiny dots)
    explored_x = [node.x for node in nodes]
    explored_y = [node.y for node in nodes]
    ax.scatter(explored_x, explored_y, color='red', s=5)  # s=5 for tiny dots

    # Plot the final path
    # Plot the final path with intermediate steps
    for i in range(len(final_path) - 1):
        x1, y1 = final_path[i]
        x2, y2 = final_path[i + 1]
        steps = interpolate_path(x1, y1, x2, y2)
        step_x, step_y = zip(*steps)
        ax.plot(step_x, step_y, color='blue', linewidth=2)

    path_x, path_y = zip(*final_path)


    ax.scatter(path_x, path_y, color='cyan', s=20, zorder=5, label='Path Points')  # larger cyan dots
    #for i, (x, y) in enumerate(zip(path_x, path_y)):
        #ax.text(x + 1, y + 1, str(i), fontsize=6, color='black')  # small number label

    ax.legend(loc='upper left')

    ax.set_title("RRT Path and Tree")

    # Get grid size
    height, width = grid.shape

    # Major ticks every 10 (with labels)
    ax.set_xticks(np.arange(0, width + 1, 10))
    ax.set_yticks(np.arange(0, height + 1, 10))

    # Minor ticks every 1 (for grid lines)
    ax.set_xticks(np.arange(0, width + 1, 1), minor=True)
    ax.set_yticks(np.arange(0, height + 1, 1), minor=True)

    # Show grid for both major and minor ticks
    ax.grid(which='minor', color='gray', linestyle='--', linewidth=0.5)
    ax.grid(which='major', color='black', linestyle='-', linewidth=0.8)


    ax.set_aspect('equal')
    plt.axis('on')  # Turn on axis if you want to see tick labels
    plt.savefig("rrt_path.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    
    '''
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
    '''