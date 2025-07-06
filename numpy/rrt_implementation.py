import numpy as np
import matplotlib.pyplot as plt
import math
import random
import heapq

# Node class representing each RRT vertex
class Node:
    def __init__(self, x, y):
        self.x = x + 0.5  # Shift to center of grid cell
        self.y = y + 0.5
        self.parent = None  # Used to trace the path back

# Define grid size and obstacle layout
grid_size = 100
grid = np.ones((grid_size, grid_size), dtype=np.uint8) * 255

# Add obstacles to the grid
grid[10:30, 40:60] = 0
grid[70:90, 20:40] = 0
grid[30:50, 70:90] = 0
grid[40:60, 50:70] = 0
grid[80:100, 80:100] = 0

# Grid to track visible areas and pheromones
visible_grid = np.zeros_like(grid, dtype=np.uint8)
pheromone_grid = np.zeros_like(grid, dtype=np.float32)

# Returns the value of the grid cell corresponding to (x, y)
def grid_value(x, y):
    ix = int(math.floor(x))
    iy = int(math.floor(y))
    if 0 <= iy < grid.shape[0] and 0 <= ix < grid.shape[1]:
        return grid[iy, ix]
    return 0

# Computes Euclidean distance and angle between two points
def distance_angle(x1, y1, x2, y2):
    distance = math.hypot(x2 - x1, y2 - y1)
    angle = math.atan2(y2 - y1, x2 - x1)
    return (distance, angle)

# Finds the closest node in node_list to (x, y)
def nearest_node(node_list, x, y):
    return min(node_list, key=lambda node: math.hypot(node.x - x, node.y - y))

# Biased random sampling based on visible space and pheromones
def ant_random_point(goal, goal_sample_rate=0.05):
    if random.random() < goal_sample_rate:
        return goal

    candidates = np.argwhere(visible_grid == 1)
    if candidates.size == 0:
        return random.randint(0, grid_size - 1), random.randint(0, grid_size - 1)

    pheromones = pheromone_grid[candidates[:, 0], candidates[:, 1]]
    probs = pheromones / pheromones.sum() if pheromones.sum() > 0 else None
    idx = np.random.choice(len(candidates), p=probs) if probs is not None else random.choice(range(len(candidates)))
    return tuple(candidates[idx][::-1])

# Moves from from_node toward (to_x, to_y) while limiting step size
def steer(from_node, to_x, to_y, step_size):
    distance, angle = distance_angle(from_node.x, from_node.y, to_x + 0.5, to_y + 0.5)
    step = min(step_size, distance)

    for i in range(1, int(step) + 1):
        new_x = from_node.x + i * math.cos(angle)
        new_y = from_node.y + i * math.sin(angle)
        if grid_value(new_x, new_y) == 0:
            return None
    return (from_node.x + step * math.cos(angle), from_node.y + step * math.sin(angle))

# Updates visibility from a given position by casting rays in all directions
def update_visibility(x, y, max_distance=grid_size):
    num_rays = 360
    for angle in np.linspace(0, 2 * np.pi, num_rays, endpoint=False):
        for dist in range(1, max_distance):
            dx = x + dist * math.cos(angle)
            dy = y + dist * math.sin(angle)
            ix = int(math.floor(dx))
            iy = int(math.floor(dy))
            if 0 <= ix < grid.shape[1] and 0 <= iy < grid.shape[0]:
                visible_grid[iy, ix] = 1
                if grid[iy, ix] == 0:
                    break
            else:
                break

# Dijkstra-based interpolation to find a grid-aligned path between two real-valued points
def interpolate_path(x1, y1, x2, y2, grid):
    # Convert float coordinates to grid indices
    x1, y1 = int(math.floor(x1)), int(math.floor(y1))
    x2, y2 = int(math.floor(x2)), int(math.floor(y2))

    # Early exit if start or goal is on obstacle
    if grid[y1, x1] == 0 or grid[y2, x2] == 0:
        return []

    height, width = grid.shape
    start = (x1, y1)
    goal = (x2, y2)

    visited = set()
    parent = {}
    cost = {start: 0}
    heap = [(0, start)]  # Priority queue for Dijkstra

    # 8-connected neighborhood
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1),
                  (-1, -1), (-1, 1), (1, -1), (1, 1)]

    while heap:
        curr_cost, current = heapq.heappop(heap)
        if current in visited:
            continue
        visited.add(current)

        if current == goal:
            break

        # Check all neighbors
        for dx, dy in directions:
            nx, ny = current[0] + dx, current[1] + dy
            if 0 <= nx < width and 0 <= ny < height and grid[ny, nx] != 0:
                next_node = (nx, ny)
                new_cost = cost[current] + 1
                if next_node not in cost or new_cost < cost[next_node]:
                    cost[next_node] = new_cost
                    parent[next_node] = current
                    heapq.heappush(heap, (new_cost, next_node))

    if goal not in parent:
        return []  # No path found

    # Reconstruct path
    path = []
    node = goal
    while node != start:
        path.append(node)
        node = parent[node]
    path.append(start)
    path.reverse()
    return path

# Checks for collision between two points using path interpolation
def collision(x1, y1, x2, y2, grid):
    return len(interpolate_path(x1, y1, x2, y2, grid)) == 0

# Draws the RRT tree
def draw_tree(nodes, ax, grid):
    for node in nodes:
        if node.parent:
            steps = interpolate_path(node.parent.x, node.parent.y, node.x, node.y, grid)
            if not steps:
                continue
            step_x, step_y = zip(*[(x + 0.5, y + 0.5) for x, y in steps])
            ax.plot(step_x, step_y, color='orange', linewidth=0.5)
            ax.scatter(step_x, step_y, color='orange', s=1)

# Extracts the final path from start to goal
def path(goal_node):
    path = []
    node = goal_node
    while node:
        path.append((node.x, node.y))
        node = node.parent
    return path[::-1]

# RRT algorithm main function
def rrt(grid, start, goal, step_size=10, max_iter=1000):
    global pheromone_grid, visible_grid

    start_node = Node(*start)  # Create starting node
    goal_node = Node(*goal)    # Create goal node

    update_visibility(start_node.x, start_node.y)  # Update what the agent can see
    nodes = [start_node]  # List of nodes in the RRT

    visited = set([(int(start_node.x), int(start_node.y))])  # Keep track of visited grid cells

    for _ in range(max_iter):
        # Sample a new point using ant-inspired bias
        rand_x, rand_y = ant_random_point(goal)

        # Find nearest node to that point
        nearest = nearest_node(nodes, rand_x + 0.5, rand_y + 0.5)

        # Try to move toward sampled point
        steered = steer(nearest, rand_x, rand_y, step_size)

        if steered:
            new_x, new_y = steered
            ix, iy = int(math.floor(new_x)), int(math.floor(new_y))

            # If not visited and path is collision-free
            if (ix, iy) not in visited and not collision(nearest.x, nearest.y, new_x, new_y, grid):
                new_node = Node(ix, iy)
                new_node.x, new_node.y = new_x, new_y  # Use exact float position
                new_node.parent = nearest
                nodes.append(new_node)
                visited.add((ix, iy))

                update_visibility(new_x, new_y)  # Reveal new area
                pheromone_grid *= 0.998  # Evaporate pheromones

                # Check if goal is reached (within distance threshold)
                if abs(new_x - goal_node.x) + abs(new_y - goal_node.y) <= 3:
                    goal_node.parent = new_node
                    nodes.append(goal_node)
                    print("Path found!")
                    return nodes

    print("Path not found")
    return None

# Deposit pheromones along the path
def deposit_pheromones(path, amount=1.0):
    for x, y in path:
        ix, iy = int(math.floor(x)), int(math.floor(y))
        if 0 <= iy < grid.shape[0] and 0 <= ix < grid.shape[1]:
            pheromone_grid[iy, ix] += amount

# Initial positions
start = (5, 5)
goal = (75, 75)

if grid[start[1], start[0]] == 0 or grid[goal[1], goal[0]] == 0:
    raise ValueError("Start or goal is inside an obstacle.")

nodes = rrt(grid, start, goal, step_size=1, max_iter=5000)

# Visualization
if nodes:
    final_path = path(nodes[-1])
    deposit_pheromones(final_path, amount=2.5)
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(grid, cmap='gray', alpha=0.8)
    draw_tree(nodes, ax, grid)

    path_x, path_y = zip(*final_path)
    ax.scatter(path_x, path_y, color='cyan', s=20, zorder=5, label='Path Points')

    for i in range(len(final_path) - 1):
        x1, y1 = final_path[i]
        x2, y2 = final_path[i + 1]
        steps = interpolate_path(x1, y1, x2, y2, grid)
        if steps:
            sx, sy = zip(*[(x + 0.5, y + 0.5) for x, y in steps])
            ax.plot(sx, sy, color='blue', linewidth=2)

    ax.legend(loc='upper left')
    ax.set_title("RRT Path and Tree")
    ax.set_xticks(np.arange(0, grid.shape[1] + 1, 10))
    ax.set_yticks(np.arange(0, grid.shape[0] + 1, 10))
    ax.set_xticks(np.arange(0, grid.shape[1] + 1, 1), minor=True)
    ax.set_yticks(np.arange(0, grid.shape[0] + 1, 1), minor=True)
    ax.grid(which='minor', color='gray', linestyle='--', linewidth=0.5)
    ax.grid(which='major', color='black', linestyle='-', linewidth=0.8)
    ax.set_aspect('equal')
    plt.savefig("aco_centered_nodes.png", dpi=300, bbox_inches='tight')
    plt.show()
