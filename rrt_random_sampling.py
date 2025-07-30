import cv2
import numpy as np
import matplotlib.pyplot as plt
import math
import random
import heapq
import json
import time

# Set seed for reproducible results
SEED = 100  # You can change this to any number
random.seed(SEED)
np.random.seed(SEED)
print(f"Using seed: {SEED}")

#load grid from image
map_img = cv2.imread("IMG_8864.png", cv2.IMREAD_GRAYSCALE)
map_img = cv2.resize(map_img, (50, 50), interpolation=cv2.INTER_NEAREST)
grid = np.where(map_img > 127, 255, 0).astype(np.uint8)
grid_size = grid.shape[0]

# Load visibility map from JSON
with open('FILE_3036.json', 'r') as f:
    visibility_map = json.load(f)

# Convert JSON coordinates to set of walkable and blocked cells
all_cells = set(map(tuple, visibility_map["all"]))      # All white cells (navigable)
blocked_cells = set(map(tuple, visibility_map["blocked"]))  # All black cells (obstacles)

# Derive walkable white cells by subtracting blocked cells
walkable_cells = all_cells - blocked_cells


# Node class representing each RRT vertex
class Node:
    def __init__(self, x, y):
        self.x = x + 0.5  # Shift to center of grid cell
        self.y = y + 0.5
        self.parent = None  # Used to trace the path back

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
def ant_random_point(goal, current_node, edge_of_coverage, goal_sample_rate=0.05):
    # With a small chance, sample the goal directly
    if random.random() < goal_sample_rate:
        return goal

    if not edge_of_coverage:
        edge_of_coverage = compute_edge_of_coverage(grid, visible_grid, walkable_cells)
        if not edge_of_coverage:
            return random.randint(0, grid_size - 1), random.randint(0, grid_size - 1)

    # Just pheromone-biased sampling from edge of coverage
    edge_list = list(edge_of_coverage)
    weights = [pheromone_grid[y, x] + 1e-6 for (x, y) in edge_list]
    total = sum(weights)

    if total <= 0 or not np.isfinite(total):
        return random.choice(edge_list)

    probs = np.array(weights) / total
    sampled_index = np.random.choice(len(edge_list), p=probs)
    return edge_list[sampled_index]

'''
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
'''
def steer(from_node, to_x, to_y, step_size):
    distance, angle = distance_angle(from_node.x, from_node.y, to_x + 0.5, to_y + 0.5)
    step = min(step_size, distance)

    for i in np.linspace(1, step, int(step) + 1):
        new_x = from_node.x + i * math.cos(angle)
        new_y = from_node.y + i * math.sin(angle)
        val = grid_value(new_x, new_y)
        if val == 0:
            #print(f"steer: Collision at ({new_x:.2f}, {new_y:.2f})")
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
def rrt(grid, start, goal, step_size=10, max_iter=1000, direct_target=False):
    global pheromone_grid, visible_grid
    global visited

    start_node = Node(*start)  # Create starting node
    goal_node = Node(*goal)    # Create goal node

    update_visibility(start_node.x, start_node.y)  # Update what the agent can see
    nodes = [start_node]  # List of nodes in the RRT

    visited = set([(int(start_node.x), int(start_node.y))])  # Keep track of visited grid cells

    for _ in range(max_iter):
        edge_of_coverage = compute_edge_of_coverage(grid, visible_grid, walkable_cells)

        # Sample point based on visibility and ACO-inspired logic
        if direct_target:
            rand_x, rand_y = goal  # force direct sampling
        else:
            rand_x, rand_y = ant_random_point(goal, nodes[-1], edge_of_coverage)
                
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
                    return nodes, len(nodes)

    print("Path not found")
    return None, 0


# Deposit pheromones along the path
def deposit_pheromones(path, amount=1.0):
    for x, y in path:
        ix, iy = int(math.floor(x)), int(math.floor(y))
        if 0 <= iy < grid.shape[0] and 0 <= ix < grid.shape[1]:
            pheromone_grid[iy, ix] += amount

#opencv visualization
def draw_result_on_image(grid, nodes, path_points, filename="rrt_output_3.png"):
    # Convert grayscale grid to BGR image
    img = np.stack([grid] * 3, axis=-1)
    img[grid == 255] = [255, 255, 255]
    img[grid == 0] = [0, 0, 0]

    scale = 15  # upscale factor for clarity
    img = cv2.resize(img, (grid.shape[1] * scale, grid.shape[0] * scale), interpolation=cv2.INTER_NEAREST)

    # === Draw full tree: orange lines between node and parent ===
    for node in nodes:
        if node.parent:
            x1, y1 = int(node.parent.x * scale), int(node.parent.y * scale)
            x2, y2 = int(node.x * scale), int(node.y * scale)
            cv2.line(img, (x1, y1), (x2, y2), (0, 165, 255), 1)  # orange

    # === Draw final path: thicker blue lines ===
    for i in range(len(path_points) - 1):
        x1, y1 = path_points[i]
        x2, y2 = path_points[i + 1]
        cx1, cy1 = int(x1 * scale), int(y1 * scale)
        cx2, cy2 = int(x2 * scale), int(y2 * scale)
        cv2.line(img, (cx1, cy1), (cx2, cy2), (255, 0, 0), 2)  # blue

    # Optional: draw circles at path points for clarity
    for (x, y) in path_points:
        cx, cy = int(x * scale), int(y * scale)
        cv2.circle(img, (cx, cy), 2, (255, 0, 0), -1)

    cv2.imwrite(filename, img)
    print(f"Image saved: {filename}")

def compute_edge_of_coverage(grid, visible_grid, walkable_cells):
    height, width = grid.shape
    seen_white = set()
    edge_candidates = set()
    unseen_white = set()

    # Step 1: Identify which walkable cells are currently visible
    for (x, y) in walkable_cells:
        if visible_grid[y, x] == 1:
            seen_white.add((x, y))  # Explored white cell
        else:
            unseen_white.add((x, y))  # Unexplored white cell

    # Step 2: Look at 4-connected neighbors of seen white cells
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    for (x, y) in seen_white:
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if (nx, ny) in walkable_cells and visible_grid[ny, nx] == 0:
                # Neighbor is white, walkable, and unseen — candidate edge
                edge_candidates.add((nx, ny))

    # Step 3: Finalize the edge by intersecting with unseen white cells
    edge_of_coverage = edge_candidates & unseen_white
    return edge_of_coverage

'''
start = (8, 21)
goal = (30, 16)

if grid[start[1], start[0]] == 0 or grid[goal[1], goal[0]] == 0:
    raise ValueError("Start or goal is inside an obstacle.")

# Run RRT and get both result and step count
nodes, length = rrt(grid, start, goal, step_size=1, max_iter=5000)




# Report results
if nodes:   
    final_path = path(nodes[-1])
    draw_result_on_image(grid, nodes, final_path)
'''

# Ensure same start and goal for comparison
start = (8, 21)
goal = (30, 16)
print(f"Start: {start}, Goal: {goal}")

if grid[start[1], start[0]] == 0 or grid[goal[1], goal[0]] == 0:
    raise ValueError("Start or goal is inside an obstacle.")

# Run RRT and get both result and step count
print("Running original RRT with ant-colony sampling...")
start_time = time.time()

nodes, length = rrt(grid, start, goal, step_size=1, max_iter=5000)

end_time = time.time()
execution_time = end_time - start_time

# Report results
if nodes:   
    final_path = path(nodes[-1])
    print(f"SUCCESS - Path found!")
    print(f"Number of RRT nodes: {len(nodes)}")
    print(f"Path length: {len(final_path)} points")
    print(f"Execution time: {execution_time:.2f} seconds")
    #draw_result_on_image(grid, nodes, final_path, "original_rrt_result.png")
else:
    print(f"FAILED - No path found in 5000 iterations")
    print(f"Execution time: {execution_time:.2f} seconds")
