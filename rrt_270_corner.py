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

def find_270_corners(grid):
    """
    Find actual 270-degree corners: outward-pointing corners of black obstacles.
    Returns the white cells diagonal to the corner black pixels.
    """
    corners = []
    h, w = grid.shape
    
    # Define the 4 possible L-shaped patterns for obstacle corners
    # Each pattern has 2 perpendicular directions that should be black (obstacle)
    corner_patterns = [
        ('up', 'right'),    # L-shape: black up and right, diagonal is down-left
        ('right', 'down'),  # L-shape: black right and down, diagonal is up-left  
        ('down', 'left'),   # L-shape: black down and left, diagonal is up-right
        ('left', 'up')      # L-shape: black left and up, diagonal is down-right
    ]
    
    # Map each pattern to its diagonal direction
    diagonal_map = {
        ('up', 'right'): (-1, 1),    # down-left
        ('right', 'down'): (-1, -1), # up-left
        ('down', 'left'): (1, -1),   # up-right  
        ('left', 'up'): (1, 1)       # down-right
    }
    
    for y in range(1, h - 1):
        for x in range(1, w - 1):
            # Only check BLACK obstacle pixels
            if grid[y, x] != 0:
                continue

            neighbors = {
                "up": grid[y - 1, x],
                "down": grid[y + 1, x],
                "left": grid[y, x - 1],
                "right": grid[y, x + 1]
            }

            # Check each L-shaped corner pattern
            for black_dirs in corner_patterns:
                # For a 270-degree corner, we need:
                # 1. The two perpendicular directions to be black (part of obstacle)
                # 2. The other two directions to be white (free space)
                other_dirs = [d for d in ['up', 'down', 'left', 'right'] if d not in black_dirs]
                
                black_condition = all(neighbors[d] == 0 for d in black_dirs)
                white_condition = all(neighbors[d] == 255 for d in other_dirs)
                
                if black_condition and white_condition:
                    # Get the diagonal offset for this pattern
                    dx, dy = diagonal_map[black_dirs]
                    corner_x, corner_y = x + dx, y + dy
                    
                    # Check if the diagonal cell is within bounds and is white
                    if (0 <= corner_x < w and 0 <= corner_y < h and 
                        grid[corner_y, corner_x] == 255):
                        corners.append((corner_x, corner_y))
                    break  # Found a corner, no need to check other patterns
    
    return corners

def visualize_corners(grid, corners, filename="270_corners_fixed.png"):
    """
    Visualize the detected corners on the grid.
    """
    # Create RGB image
    img = np.stack([grid] * 3, axis=-1)
    img[grid == 255] = [255, 255, 255]  # white for free space
    img[grid == 0] = [0, 0, 0]          # black for obstacles

    # Mark corners in red
    for x, y in corners:
        img[y, x] = [255, 0, 0]  # red for corners
        

    # Resize for better visibility
    img = cv2.resize(img, (500, 500), interpolation=cv2.INTER_NEAREST)
    cv2.imwrite(filename, img)
    print(f"Saved corner visualization to: {filename}")
    return img

# Node class representing each RRT vertex
class Node:
    def __init__(self, x, y):
        self.x = x + 0.5  # Shift to center of grid cell
        self.y = y + 0.5
        self.parent = None  # Used to trace the path back

# Grid to track visible areas and pheromones (will be initialized in main)
visible_grid = None
pheromone_grid = None

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

def is_target_visible(current_pos, target_pos, grid, max_distance=None):
    """
    Check if target is visible from current position using ray casting.
    Returns True if there's a clear line of sight.
    """
    x1, y1 = current_pos
    x2, y2 = target_pos
    
    # Calculate distance and angle
    distance = math.hypot(x2 - x1, y2 - y1)
    if max_distance and distance > max_distance:
        return False
    
    angle = math.atan2(y2 - y1, x2 - x1)
    
    # Cast ray from current position to target
    steps = int(distance) + 1
    for i in range(1, steps):
        t = i / steps
        ray_x = x1 + t * (x2 - x1)
        ray_y = y1 + t * (y2 - y1)
        
        if grid_value(ray_x, ray_y) == 0:  # Hit obstacle
            return False
    
    return True

def get_visible_corners(current_pos, corner_list, grid, max_distance=None):
    """
    Get list of 270-degree corners that are visible from current position.
    """
    visible_corners = []
    
    for corner in corner_list:
        corner_x, corner_y = corner
        if is_target_visible(current_pos, (corner_x + 0.5, corner_y + 0.5), grid, max_distance):
            visible_corners.append(corner)
    
    return visible_corners

def compute_corner_visibility_set(corner_pos, grid, walkable_cells, max_distance=None):
    """
    Compute the set of walkable cells visible from a given corner position.
    Uses ray casting in all directions.
    """
    if max_distance is None:
        max_distance = max(grid.shape)
    
    corner_x, corner_y = corner_pos
    cx, cy = corner_x + 0.5, corner_y + 0.5  # Center of corner cell
    
    visible_set = set()
    num_rays = 360  # Cast rays in all directions
    
    for angle in np.linspace(0, 2 * np.pi, num_rays, endpoint=False):
        for dist in range(1, max_distance):
            ray_x = cx + dist * math.cos(angle)
            ray_y = cy + dist * math.sin(angle)
            
            # Convert to grid coordinates
            grid_x = int(math.floor(ray_x))
            grid_y = int(math.floor(ray_y))
            
            # Check bounds
            if not (0 <= grid_x < grid.shape[1] and 0 <= grid_y < grid.shape[0]):
                break
            
            # If this cell is walkable, add it to visible set
            if (grid_x, grid_y) in walkable_cells:
                visible_set.add((grid_x, grid_y))
            
            # If we hit an obstacle, stop this ray
            if grid[grid_y, grid_x] == 0:
                break
    
    return visible_set

def select_best_corner(visible_corners, edge_of_coverage, grid, walkable_cells):
    """
    Select the corner that has maximum intersection with edge of coverage.
    Returns the best corner and its intersection size.
    """
    if not visible_corners:
        return None, 0
    
    best_corner = None
    max_intersection = 0
    
    for corner in visible_corners:
        # Compute visibility set for this corner
        corner_visibility = compute_corner_visibility_set(corner, grid, walkable_cells)
        
        # Find intersection with edge of coverage
        intersection = corner_visibility & edge_of_coverage
        intersection_size = len(intersection)
        
        if intersection_size > max_intersection:
            max_intersection = intersection_size
            best_corner = corner
    
    return best_corner, max_intersection

# Enhanced biased random sampling using 270-degree corners
def corner_based_sampling(goal, current_node, edge_of_coverage, corner_list, grid, walkable_cells, goal_sample_rate=0.05):
    """
    Enhanced sampling strategy using 270-degree corners.
    """
    current_pos = (current_node.x, current_node.y)
    
    # First check: Can we see the target directly?
    if is_target_visible(current_pos, goal, grid):
        print("Target visible! Sampling towards target.")
        return goal[0] - 0.5, goal[1] - 0.5  # Convert to grid coordinates
    
    # If no edge of coverage, fall back to random sampling
    if not edge_of_coverage:
        return random.randint(0, grid.shape[1] - 1), random.randint(0, grid.shape[0] - 1)
    
    # Get visible 270-degree corners
    visible_corners = get_visible_corners(current_pos, corner_list, grid)
    
    if not visible_corners:
        # No corners visible, sample from edge of coverage
        print("No corners visible, sampling from edge of coverage.")
        return random.choice(list(edge_of_coverage))
    
    # Select the best corner based on edge coverage intersection
    best_corner, intersection_size = select_best_corner(visible_corners, edge_of_coverage, grid, walkable_cells)
    
    if best_corner and intersection_size > 0:
        print(f"Selected corner {best_corner} with {intersection_size} edge intersections.")
        return best_corner
    else:
        # Fallback to random corner if no good intersection found
        print("No good corner intersection, selecting random visible corner.")
        return random.choice(visible_corners)

def steer(from_node, to_x, to_y, step_size):
    distance, angle = distance_angle(from_node.x, from_node.y, to_x + 0.5, to_y + 0.5)
    step = min(step_size, distance)

    for i in np.linspace(1, step, int(step) + 1):
        new_x = from_node.x + i * math.cos(angle)
        new_y = from_node.y + i * math.sin(angle)
        val = grid_value(new_x, new_y)
        if val == 0:
            return None
    return (from_node.x + step * math.cos(angle), from_node.y + step * math.sin(angle))

# Updates visibility from a given position by casting rays in all directions
def update_visibility(x, y, max_distance=None):
    if max_distance is None:
        max_distance = max(grid.shape)
    
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

# Enhanced RRT algorithm with corner-based sampling
def enhanced_rrt(grid, start, goal, corner_list, walkable_cells, step_size=10, max_iter=1000):
    global pheromone_grid, visible_grid
    
    # Initialize global grids if not already done
    if visible_grid is None:
        visible_grid = np.zeros_like(grid, dtype=np.uint8)
    if pheromone_grid is None:
        pheromone_grid = np.zeros_like(grid, dtype=np.float32)

    start_node = Node(*start)  # Create starting node
    goal_node = Node(*goal)    # Create goal node

    update_visibility(start_node.x, start_node.y)  # Update what the agent can see
    nodes = [start_node]  # List of nodes in the RRT

    visited = set([(int(start_node.x), int(start_node.y))])  # Keep track of visited grid cells

    for iteration in range(max_iter):
        if iteration % 100 == 0:
            print(f"Iteration {iteration}: {len(nodes)} nodes")
        
        edge_of_coverage = compute_edge_of_coverage(grid, visible_grid, walkable_cells)

        # Use enhanced corner-based sampling
        rand_x, rand_y = corner_based_sampling(
            (goal_node.x, goal_node.y), 
            nodes[-1], 
            edge_of_coverage, 
            corner_list, 
            grid, 
            walkable_cells
        )
                
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

#opencv visualization
def draw_result_on_image(grid, nodes, path_points, filename="rrt_output_enhanced.png"):
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

# Main execution
if __name__ == "__main__":
    # Load grid from image
    map_img = cv2.imread("IMG_8864.png", cv2.IMREAD_GRAYSCALE)
    map_img = cv2.resize(map_img, (50, 50), interpolation=cv2.INTER_NEAREST)
    grid = np.where(map_img > 127, 255, 0).astype(np.uint8)
    grid_size = grid.shape[0]

    # Initialize global grids
    visible_grid = np.zeros_like(grid, dtype=np.uint8)
    pheromone_grid = np.zeros_like(grid, dtype=np.float32)

    # Find 270-degree corners
    corners_270 = find_270_corners(grid)
    print(f"Found {len(corners_270)} 270-degree corners")
    visualize_corners(grid, corners_270, "corners_enhanced.png")

    # Load visibility map from JSON
    with open('FILE_3036.json', 'r') as f:
        visibility_map = json.load(f)

    # Convert JSON coordinates to set of walkable and blocked cells
    all_cells = set(map(tuple, visibility_map["all"]))
    blocked_cells = set(map(tuple, visibility_map["blocked"]))
    walkable_cells = all_cells - blocked_cells
    '''
    # Define start and goal
    start = (8, 21)
    goal = (30, 16)

    if grid[start[1], start[0]] == 0 or grid[goal[1], goal[0]] == 0:
        raise ValueError("Start or goal is inside an obstacle.")

    # Run enhanced RRT with corner-based sampling
    print("Running enhanced RRT with 270-degree corner sampling...")
    nodes, length = enhanced_rrt(
        grid, start, goal, corners_270, walkable_cells, 
        step_size=1, max_iter=5000
    )
    
    # Report results
    if nodes:   
        final_path = path(nodes[-1])
        print(f"Path found with {len(final_path)} points")
        draw_result_on_image(grid, nodes, final_path, "enhanced_rrt_result.png")
    else:
        print("No path found")
    '''

    # Ensure same start and goal for comparison
    start = (8, 21)
    goal = (30, 16)
    print(f"Start: {start}, Goal: {goal}")
    
    # Add before calling RRT
    start_time = time.time()


    # For the enhanced file, modify to:
    nodes, length = enhanced_rrt(grid, start, goal, corners_270, walkable_cells, step_size=1, max_iter=5000)

    # Add after RRT call in both files:
    end_time = time.time()
    execution_time = end_time - start_time

    if nodes:
        final_path = path(nodes[-1])
        print(f"SUCCESS - Path found!")
        print(f"Number of RRT nodes: {len(nodes)}")
        print(f"Path length: {len(final_path)} points")
        print(f"Execution time: {execution_time:.2f} seconds")
    else:
        print(f"FAILED - No path found in 5000 iterations")
        print(f"Execution time: {execution_time:.2f} seconds")