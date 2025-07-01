import numpy as np
import matplotlib.pyplot as plt
import math
import random
import heapq

class Node:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.parent = None


grid_size = 100 #change this to liking
grid = np.ones((grid_size, grid_size), dtype=np.uint8) * 255

# rectangle obstacles

grid[10:30, 40:60] = 0
grid[70:90, 20:40] = 0
grid[30:50, 70:90] = 0
grid[40:60, 50:70] = 0
grid[80:100, 80:100] = 0
'''

#maze-like obstacles
grid[0:70, 20:25] = 0
grid[30:100, 65:70] = 0
'''
# For Partial observability and pheromone tracking
visible_grid = np.zeros_like(grid, dtype=np.uint8)  # 0 = unknown, 1 = seen
pheromone_grid = np.zeros_like(grid, dtype=np.float32)  # stores pheromone levels


def distance_angle(x1, y1, x2, y2):
    distance = math.sqrt(((x1 - x2)**2) + ((y1 - y2)**2))
    dy = y2 - y1
    dx = x2 - x1
    angle = math.atan2(dy, dx)
    return (distance, angle)

def nearest_node(node_list, x, y):
    closest_node = None  # Initialize closest node as None
    min_distance = float('inf')  # Set minimum distance to a large value

    for node in node_list:  # Iterate through all nodes
        distance = math.sqrt(((node.x - x)**2) + ((node.y - y)**2))  # Compute distance

        if distance < min_distance:  # If a closer node is found
            min_distance = distance  # Update minimum distance
            closest_node = node  # Update closest node

    return closest_node  # Return the nearest node

#Bias sampling based on visible world and pheromone trails 
def ant_random_point(goal, goal_sample_rate=0.05):
    if random.random() < goal_sample_rate:
        return goal  # Only bias if goal is seen

    candidates = np.argwhere(visible_grid == 1)
    if candidates.size == 0:
        return random.randint(0, grid_size - 1), random.randint(0, grid_size - 1)

    pheromones = pheromone_grid[candidates[:, 0], candidates[:, 1]]
    if pheromones.sum() == 0:
        idx = random.choice(range(len(candidates)))
    else:
        probs = pheromones / pheromones.sum()
        idx = np.random.choice(len(candidates), p=probs)

    return tuple(candidates[idx][::-1])


#moves toward target while limitting step size
def steer(from_node, to_x, to_y, step_size):
    distance, angle = distance_angle(from_node.x, from_node.y, to_x, to_y)
    step = min(step_size, distance)

    for i in range(1, int(step) + 1):
        new_x = int(round(from_node.x + i * math.cos(angle)))
        new_y = int(round(from_node.y + i * math.sin(angle)))

        # Bounds check
        if not (0 <= new_x < grid.shape[1] and 0 <= new_y < grid.shape[0]):
            return None
        
        # Obstacle check
        if grid[new_y, new_x] == 0:
            return None
        
        # Early return if we're at max step or near goal
        if i == int(step):
            return (new_x, new_y)

    return None  # Fallback if nothing valid found


def update_visibility(x, y, radius=6):
    for dx in range(-radius, radius + 1):
        for dy in range(-radius, radius + 1):
            nx, ny = x + dx, y + dy
            if 0 <= nx < grid.shape[1] and 0 <= ny < grid.shape[0]:
                if grid[ny, nx] != 0:
                    visible_grid[ny, nx] = 1


#main function
def rrt(grid, start, goal, step_size=10, max_iter=1000):
    global pheromone_grid, visible_grid
    visited = set()  # Track visited nodes

    start_node = Node(start[0], start[1])
    update_visibility(start_node.x, start_node.y, radius=15)
    goal_node = Node(goal[0], goal[1])
    visited.add((start_node.x, start_node.y))
    nodes = [start_node]

    for _ in range(max_iter):
        rand_x, rand_y = ant_random_point(goal) # Generate random point
        nearest = nearest_node(nodes, rand_x, rand_y)  # Find nearest node
        steered = steer(nearest, rand_x, rand_y, step_size)  # Move toward random point

        if steered:  # If a valid move is found
            new_x, new_y = steered
            if (new_x, new_y) not in visited and not collision(nearest.x, nearest.y, new_x, new_y, grid):  #if no collision
                new_node = Node(new_x, new_y)  # Create new node
                new_node.parent = nearest  # Set parent
                nodes.append(new_node)  # Add node to list
                visited.add((new_x, new_y))  # Mark as visited

                update_visibility(new_x, new_y, radius=6)  # Update local visibility after adding node
                pheromone_grid *= 0.998  # Evaporate pheromones gradually


                if abs(new_x - goal[0]) + abs(new_y - goal[1]) <= 3:  # Check if goal is reached
                    goal_node.parent = new_node  # Set goal parent
                    nodes.append(goal_node)  # Add goal node
                    print("Path found!")
                    return nodes

    print("Path not found")
    return None


def is_moving_towards_goal(current, next_node, goal):
    """
    Helper to ensure Dijkstra only progresses roughly toward the goal.
    This limits excessive branching.
    """
    dx1 = goal[0] - current[0]
    dy1 = goal[1] - current[1]
    dx2 = next_node[0] - current[0]
    dy2 = next_node[1] - current[1]

    # dot product should be positive (angle < 90 degrees) and ensures path is moving towards gaol
    return dx1 * dx2 + dy1 * dy2 > 0

def interpolate_path(x1, y1, x2, y2, grid):
    """
    Uses Dijkstra's algorithm to find a grid-aligned path that avoids obstacles.
    This ensures smooth movement between two points.
    """
    height, width = grid.shape
    start = (x1, y1)
    goal = (x2, y2)

    # Early exit if start or goal is invalid
    if grid[y1, x1] == 0 or grid[y2, x2] == 0:
        return []

    visited = set()  # Set to track visited nodes
    parent = {}  # Stores parent nodes for path reconstruction
    cost = {start: 0}  # Stores cost from start to each node (edges)
    heap = [(0, start)]  # Priority queue (min-heap) for Dijkstra's algorithm

    # Directions for 8-connectivity
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1),
                  (-1, -1), (-1, 1), (1, -1), (1, 1)]

    while heap: # Process nodes in priority queue
        curr_cost, current = heapq.heappop(heap) # Get node with lowest cost
        if current in visited:
            continue
        visited.add(current)

        if current == goal:
            break

        for dx, dy in directions: # Explore neighboring nodes
            nx, ny = current[0] + dx, current[1] + dy # Compute new coordinates

            if 0 <= nx < width and 0 <= ny < height and grid[ny, nx] != 0:
                next_node = (nx, ny)
                new_cost = cost[current] + 1 #cost to reach next node

                # Only allow movement roughly in the direction of goal
                if is_moving_towards_goal(current, next_node, goal):
                    if next_node not in cost or new_cost < cost[next_node]: # Update cost if lower
                        cost[next_node] = new_cost
                        parent[next_node] = current
                        heapq.heappush(heap, (new_cost, next_node)) # Add to priority queue

    if goal not in parent:
        return []

    # Reconstruct path from goal to start
    path = []
    node = goal
    while node != start:
        path.append(node)
        node = parent[node]

    path.append(start)
    path.reverse()

    return path


def collision(x1, y1, x2, y2, grid):
    """
    Checks if there is a valid path between two points.
    If no path is found, it means there is an obstacle blocking movement.
    """
    path = interpolate_path(x1, y1, x2, y2, grid)
    return len(path) == 0  # if no path found, there’s a collision

def draw_tree(nodes, ax, grid):
    """
    Visualizes the RRT tree by drawing connections between nodes.
    """
    for node in nodes:
        if node.parent is not None:
            steps = interpolate_path(node.parent.x, node.parent.y, node.x, node.y, grid)  # Compute path
            if not steps:  # Skip if path is invalid
                continue
            step_x, step_y = zip(*steps) # Extract x and y coordinates from path
            ax.plot(step_x, step_y, color='orange', linewidth=0.5) # Draw path as thin orange lines
            ax.scatter(step_x, step_y, color='orange', s=1)  #each step as small orange dots

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
#start= (10, 10)
#goal = (85, 85)
start = (5, 5)
goal = (75, 75)

if grid[start[1], start[0]] == 0 or grid[goal[1], goal[0]] == 0:
    raise ValueError("Start or goal is inside an obstacle.")

nodes = rrt(grid, start, goal, step_size=5, max_iter=5000)

# === NEW: Deposit pheromones on successful path ===
def deposit_pheromones(path, amount=1.0):
    for x, y in path:
        pheromone_grid[y, x] += amount

#visualize
if nodes:
    final_path = path(nodes[-1])
    deposit_pheromones(final_path, amount=2.5)


    fig, ax = plt.subplots(figsize=(10, 10))  # Larger figure
    ax.imshow(grid, cmap='gray', alpha=0.8)

    # Draw RRT tree
    draw_tree(nodes, ax, grid)

    # Scatter all explored nodes (tiny dots)
    explored_x = [node.x for node in nodes]
    explored_y = [node.y for node in nodes]
    ax.scatter(explored_x, explored_y, color='red', s=5)  # s=5 for tiny dots

    # Plot the final path
    # Plot the final path with intermediate steps
    for i in range(len(final_path) - 1):
        x1, y1 = final_path[i]
        x2, y2 = final_path[i + 1]
        steps = interpolate_path(x1, y1, x2, y2, grid)
        if not steps:
            continue
        step_x, step_y = zip(*steps)
        ax.plot(step_x, step_y, color='blue', linewidth=2)

    path_x, path_y = zip(*final_path)


    ax.scatter(path_x, path_y, color='cyan', s=20, zorder=5, label='Path Points')  # larger cyan dots

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
    plt.savefig("aco_radius_observibility_step5.png", dpi=300, bbox_inches='tight')
    plt.show()
