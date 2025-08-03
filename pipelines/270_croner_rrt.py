import cv2
import numpy as np
import matplotlib.pyplot as plt
import math
import random
import heapq
import json
import time
import os
from pathlib import Path

# Global grids
visible_grid = None
pheromone_grid = None
grid = None
walkable_cells = None
corner_list = None
#Deborah's version

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

#Edwin's version
'''
def find_270_corners(grid):
    """
    Find 270-degree corners using 2x2 block analysis.
    Grid format: 255 = free space, 0 = obstacles
    """
    corners = []
    rows, cols = grid.shape
    
    # Convert to binary format for the 2x2 analysis
    # 0 = free space, 1 = obstacles (opposite of your grid format)
    #binary_grid = np.where(grid == 255, 0, 1)
    binary_grid = np.where(grid > 128, 0, 1)
    
    for i in range(rows - 1):
        for j in range(cols - 1):
            block = binary_grid[i:i+2, j:j+2]
            # Check if exactly 3 cells are free (sum == 3) and 1 cell is obstacle
            if block.sum() == 3:
                # Find the obstacle cell position within the 2x2 block
                obstacle_idx = np.argwhere(block == 0)[0]
                # Calculate the diagonal position (the free cell diagonal to the obstacle)
                di, dj = 1 - obstacle_idx[0], 1 - obstacle_idx[1]
                corner_i, corner_j = i + di, j + dj
                corners.append((corner_i, corner_j))  # Return as (x, y) to match your existing format
    
    return corners
'''


# Node class representing each RRT vertex
class Node:
    def __init__(self, x, y):
        self.x = x + 0.5  # Shift to center of grid cell
        self.y = y + 0.5
        self.parent = None  # Used to trace the path back

# Returns the value of the grid cell corresponding to (x, y)
def grid_value(x, y):
    global grid
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
    Returns (x, y, is_corner) where is_corner indicates if this is a 270-degree corner.
    """
    current_pos = (current_node.x, current_node.y)
    
    # First check: Can we see the target directly?
    if is_target_visible(current_pos, goal, grid):
        return goal[0] - 0.5, goal[1] - 0.5, False  # Convert to grid coordinates
    
    # If no edge of coverage, fall back to random sampling
    if not edge_of_coverage:
        return random.randint(0, grid.shape[1] - 1), random.randint(0, grid.shape[0] - 1), False
    
    # Get visible 270-degree corners
    visible_corners = get_visible_corners(current_pos, corner_list, grid)
    
    if not visible_corners:
        # No corners visible, sample from edge of coverage
        edge_cell = random.choice(list(edge_of_coverage))
        return edge_cell[0], edge_cell[1], False
    
    # Select the best corner based on edge coverage intersection
    best_corner, intersection_size = select_best_corner(visible_corners, edge_of_coverage, grid, walkable_cells)
    
    if best_corner and intersection_size > 0:
        return best_corner[0], best_corner[1], True  # This is a corner
    else:
        # Fallback to random corner if no good intersection found
        fallback_corner = random.choice(visible_corners)
        return fallback_corner[0], fallback_corner[1], True  # This is a corner

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

def steer_to_corner(from_node, corner_x, corner_y):
    """
    Steer directly to a corner without step size limitation.
    Returns the corner position if path is clear, None if blocked.
    """
    target_x = corner_x + 0.5  # Center of corner cell
    target_y = corner_y + 0.5
    
    # Check if path to corner is clear
    if not collision(from_node.x, from_node.y, target_x, target_y, grid):
        return (target_x, target_y)
    else:
        return None
    
# Updates visibility from a given position by casting rays in all directions
def update_visibility(x, y, max_distance=None):
    global visible_grid, grid
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

# Extracts the final path from start to goal
def path_length(goal_node):
    if goal_node is None:
        return 0.0
    length = 0.0
    node = goal_node
    while node.parent is not None:
        dx = node.x - node.parent.x
        dy = node.y - node.parent.y
        length += math.hypot(dx, dy)
        node = node.parent
    return length

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

def create_walkable_cells(grid):
    """Create set of walkable cells from grid"""
    walkable_cells = set()
    height, width = grid.shape
    for y in range(height):
        for x in range(width):
            if grid[y, x] == 255:  # White cells are walkable
                walkable_cells.add((x, y))
    return walkable_cells

# Enhanced RRT algorithm with corner-based sampling
def enhanced_rrt(grid_param, start, goal, corner_list_param, walkable_cells_param, step_size=10, max_iter=1000):
    global pheromone_grid, visible_grid, grid, walkable_cells, corner_list
    
    # Initialize global variables
    grid = grid_param
    walkable_cells = walkable_cells_param
    corner_list = corner_list_param
    visible_grid = np.zeros_like(grid, dtype=np.uint8)
    pheromone_grid = np.zeros_like(grid, dtype=np.float32)

    start_node = Node(*start)  # Create starting node
    goal_node = Node(*goal)    # Create goal node

    update_visibility(start_node.x, start_node.y)  # Update what the agent can see
    nodes = [start_node]  # List of nodes in the RRT

    visited = set([(int(start_node.x), int(start_node.y))])  # Keep track of visited grid cells

    for iteration in range(max_iter):
        edge_of_coverage = compute_edge_of_coverage(grid, visible_grid, walkable_cells)

        # Use enhanced corner-based sampling
        rand_x, rand_y, is_corner = corner_based_sampling(
            (goal_node.x, goal_node.y), 
            nodes[-1], 
            edge_of_coverage, 
            corner_list, 
            grid, 
            walkable_cells
        )
                
        # Find nearest node to that point
        nearest = nearest_node(nodes, rand_x + 0.5, rand_y + 0.5)

        # Choose steering method based on whether target is a corner
        if is_corner:
            # Go directly to the corner (no step size limit)
            steered = steer_to_corner(nearest, rand_x, rand_y)
        else:
            # Use normal step-limited steering
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
                    # Compute path-node count
                    path_nodes = []
                    curr = goal_node
                    while curr:
                        path_nodes.append(curr)
                        curr = curr.parent
                    path_node_count = len(path_nodes)
                    return nodes, len(nodes), path_node_count  # (tree nodes, total explored, nodes on final path)


    return None, 0, 0

def load_shelf_worlds(shelf_dir=".", num_worlds=60):
    """Load existing shelf worlds from .bak/.dir/.dat files, searching recursively"""
    import struct
    import glob
    
    worlds = []
    shelf_files = []
    
    # Recursively find all .dat files in subdirectories
    print(f"Searching for .dat files recursively in: {shelf_dir}")
    dat_files = []
    
    # Use Path.rglob for recursive globbing
    shelf_path = Path(shelf_dir)
    if shelf_path.exists():
        dat_files = list(shelf_path.rglob("*.dat"))
        dat_files = [str(f) for f in dat_files]  # Convert Path objects to strings
        dat_files.sort()  # Ensure consistent ordering
        print(f"Found {len(dat_files)} .dat files:")
        for f in dat_files[:10]:  # Show first 10 files found
            print(f"  {f}")
        if len(dat_files) > 10:
            print(f"  ... and {len(dat_files) - 10} more files")
    else:
        print(f"Directory {shelf_dir} does not exist!")
        return [], []
    
    if len(dat_files) == 0:
        print("No .dat files found!")
        return [], []
    
    if len(dat_files) < num_worlds:
        print(f"Warning: Found only {len(dat_files)} shelf files, using all available")
        num_worlds = len(dat_files)
    
    for i in range(num_worlds):
        dat_file = dat_files[i]
        # Create a more descriptive shelf name including the subdirectory
        relative_path = os.path.relpath(dat_file, shelf_dir)
        shelf_name = os.path.splitext(relative_path)[0].replace(os.sep, '_')
        shelf_files.append(shelf_name)
        
        try:
            # Try to load as binary data first
            with open(dat_file, 'rb') as f:
                data = f.read()
            
            print(f"Loading {dat_file} (size: {len(data)} bytes)")
            
            # Try different approaches to interpret the data
            # Approach 1: Assume it's a flattened grid
            if len(data) == 10000:  # 100x100 grid
                grid = np.frombuffer(data, dtype=np.uint8).reshape((100, 100))
            elif len(data) == 2500:  # 50x50 grid  
                grid = np.frombuffer(data, dtype=np.uint8).reshape((50, 50))
            elif len(data) == 40000:  # 100x100 grid with 4 bytes per pixel
                grid = np.frombuffer(data, dtype=np.uint32).reshape((100, 100))
                # Convert to uint8
                grid = (grid > 0).astype(np.uint8) * 255
            elif len(data) == 160000:  # 200x200 grid
                grid = np.frombuffer(data, dtype=np.uint8).reshape((200, 200))
            else:
                # Try to determine grid size from file size
                sqrt_size = int(math.sqrt(len(data)))
                if sqrt_size * sqrt_size == len(data):
                    grid = np.frombuffer(data, dtype=np.uint8).reshape((sqrt_size, sqrt_size))
                    print(f"  Detected {sqrt_size}x{sqrt_size} grid")
                else:
                    # Approach 2: Try as text file
                    try:
                        with open(dat_file, 'r') as f:
                            lines = f.readlines()
                        # Parse text-based grid format
                        grid_data = []
                        for line in lines:
                            if line.strip():
                                # Try different delimiters
                                if ',' in line:
                                    row = [int(float(x.strip())) for x in line.strip().split(',') if x.strip()]
                                else:
                                    row = [int(float(x)) for x in line.strip().split() if x.strip()]
                                if row:  # Only add non-empty rows
                                    grid_data.append(row)
                        if grid_data:
                            grid = np.array(grid_data, dtype=np.uint8)
                            print(f"  Loaded as text file: {grid.shape}")
                        else:
                            raise ValueError("No valid data found in text file")
                    except Exception as text_error:
                        # Approach 3: Create a default grid if loading fails
                        print(f"Warning: Could not load {dat_file} (text error: {text_error}), using default grid")
                        grid = np.ones((100, 100), dtype=np.uint8) * 255
                        # Add some obstacles
                        grid[30:70, 30:70] = 0
            
            # Ensure grid values are in correct format (255 for free, 0 for obstacles)
            if grid.max() <= 1:
                grid = grid * 255  # Convert from 0-1 to 0-255
            elif grid.max() > 255:
                # Normalize to 0-255 range
                grid = ((grid - grid.min()) / (grid.max() - grid.min()) * 255).astype(np.uint8)
            
            # Ensure we have some free space and some obstacles
            free_cells = np.sum(grid > 0)
            total_cells = grid.size
            free_ratio = free_cells / total_cells
            
            print(f"  Grid shape: {grid.shape}, Free space: {free_ratio:.2%}")
            
            # If grid is all zeros or all ones, create a more interesting layout
            if free_ratio < 0.1 or free_ratio > 0.95:
                print(f"  Warning: Grid has extreme free space ratio ({free_ratio:.2%}), creating balanced grid")
                grid = np.ones(grid.shape, dtype=np.uint8) * 255
                # Add some obstacles (20-30% of the grid)
                obstacle_ratio = 0.25
                num_obstacles = int(grid.size * obstacle_ratio)
                for _ in range(num_obstacles):
                    x = random.randint(0, grid.shape[1] - 1)
                    y = random.randint(0, grid.shape[0] - 1)
                    grid[y, x] = 0
            
            worlds.append(grid)
            
        except Exception as e:
            print(f"Error loading {dat_file}: {e}")
            # Create a fallback grid
            grid = np.ones((100, 100), dtype=np.uint8) * 255
            grid[20:80, 20:80] = 0  # Simple obstacle
            worlds.append(grid)
            print(f"  Created fallback grid: {grid.shape}")
    
    print(f"Successfully loaded {len(worlds)} shelf worlds from {shelf_dir}")
    return worlds, shelf_files

def generate_start_goal_pairs(grid, num_pairs=5, min_distance=30):
    """Generate 5 unique start-goal pairs for a given world"""
    pairs = []
    attempts = 0
    max_attempts = 1000
    
    # Find all free space positions
    free_positions = []
    for y in range(5, grid.shape[0] - 5):
        for x in range(5, grid.shape[1] - 5):
            if grid[y, x] != 0:  # Free space
                free_positions.append((x, y))
    
    if len(free_positions) < 2:
        print(f"  Warning: Very few free positions ({len(free_positions)}), using default positions")
        return [((10, 10), (grid.shape[1] - 10, grid.shape[0] - 10))]
    
    print(f"  Found {len(free_positions)} free positions")
    
    while len(pairs) < num_pairs and attempts < max_attempts:
        # Randomly select from free positions
        start = random.choice(free_positions)
        goal = random.choice(free_positions)
        
        start_x, start_y = start
        goal_x, goal_y = goal
        
        # Check minimum distance
        distance = math.sqrt((goal_x - start_x)**2 + (goal_y - start_y)**2)
        if distance < min_distance:
            attempts += 1
            continue
            
        # Check if this pair is too similar to existing pairs
        too_similar = False
        for existing_start, existing_goal in pairs:
            start_dist = math.sqrt((start_x - existing_start[0])**2 + (start_y - existing_start[1])**2)
            goal_dist = math.sqrt((goal_x - existing_goal[0])**2 + (goal_y - existing_goal[1])**2)
            if start_dist < 15 and goal_dist < 15:
                too_similar = True
                break
                
        if not too_similar:
            pairs.append((start, goal))
        
        attempts += 1
    
    if len(pairs) < num_pairs:
        print(f"  Warning: Could only generate {len(pairs)} pairs instead of {num_pairs}")
    
    return pairs


def run_experiment(shelf_dir="."):
    """Main experimental pipeline for 270 Corner RRT with full metrics"""
    print("=" * 60)
    print("270 CORNER RRT EXPERIMENTAL PIPELINE")
    print("=" * 60)

    results_dir = Path("corner_rrt_results")
    results_dir.mkdir(exist_ok=True)

    worlds, shelf_names = load_shelf_worlds(shelf_dir, num_worlds=60)
    if len(worlds) == 0:
        print("ERROR: No worlds loaded! Please check the directory path and file format.")
        return []

    results = []
    total_runs = 0
    successful_runs = 0

    for world_id, (grid, shelf_name) in enumerate(zip(worlds, shelf_names)):
        print(f"\nShelf {world_id + 1}/{len(worlds)}: {shelf_name}")
        print(f"  Grid size: {grid.shape}")

        # Use a fixed seed per world for reproducibility
        world_seed = 100 + world_id
        random.seed(world_seed)
        np.random.seed(world_seed)

        walkable_cells = create_walkable_cells(grid)
        print(f"  Found {len(walkable_cells)} walkable cells")

        corners_270 = find_270_corners(grid)
        print(f"  Found {len(corners_270)} 270-degree corners")

        pairs = generate_start_goal_pairs(grid, num_pairs=5)
        print(f"  Generated {len(pairs)} start-goal pairs")

        for pair_id, (start, goal) in enumerate(pairs):
            print(f"  Pair {pair_id + 1}: Start{start} -> Goal{goal}")

            # Use the same seed per pair for consistent behavior
            random.seed(world_seed)
            np.random.seed(world_seed)

            start_time = time.time()
            nodes, node_count, path_node_count = enhanced_rrt(
                grid, start, goal, corners_270, walkable_cells, step_size=2, max_iter=5000
            )
            end_time = time.time()
            execution_time = end_time - start_time
            total_runs += 1

            if nodes:
                successful_runs += 1
                path_len = path_length(nodes[-1])
                path_efficiency = path_node_count / node_count if node_count > 0 else 0
                success = True
                print(f"    SUCCESS: {node_count} nodes, {path_node_count} on final path ({path_efficiency:.1%}), {path_len:.1f} path length, {execution_time:.2f}s")
            else:
                path_len = 0
                path_node_count = 0
                path_efficiency = 0
                success = False
                print(f"    FAILED: No path found, {execution_time:.2f}s")

            results.append({
                'world_id': world_id,
                'shelf_name': shelf_name,
                'pair_id': pair_id,
                'world_seed': world_seed,
                'start': start,
                'goal': goal,
                'success': success,
                'nodes_explored': node_count,
                'nodes_on_final_path': path_node_count,
                'path_efficiency': path_efficiency,
                'path_length': path_len,
                'execution_time': execution_time,
                'algorithm': '270_corner_rrt',
                'corners_found': len(corners_270)
            })

    results_file = results_dir / "corner_rrt_results.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 60)
    print("270 CORNER RRT - FINAL RESULTS SUMMARY")
    print("=" * 60)
    print(f"Total runs: {total_runs}")
    print(f"Successful runs: {successful_runs}")

    if total_runs > 0:
        print(f"Success rate: {successful_runs/total_runs*100:.1f}%")
        if successful_runs > 0:
            successful_results = [r for r in results if r['success']]
            avg_nodes = np.mean([r['nodes_explored'] for r in successful_results])
            avg_path_nodes = np.mean([r['nodes_on_final_path'] for r in successful_results])
            avg_efficiency = np.mean([r['path_efficiency'] for r in successful_results])
            avg_path_length = np.mean([r['path_length'] for r in successful_results])
            avg_time = np.mean([r['execution_time'] for r in successful_results])
            avg_corners = np.mean([r['corners_found'] for r in results])

            print(f"Average nodes explored: {avg_nodes:.1f}")
            print(f"Average nodes on final path: {avg_path_nodes:.1f}")
            print(f"Average path efficiency: {avg_efficiency:.1%}")
            print(f"Average path length: {avg_path_length:.1f}")
            print(f"Average execution time: {avg_time:.2f}s")
            print(f"Average corners found per world: {avg_corners:.1f}")
    else:
        print("No runs completed - check if worlds were loaded correctly!")

    print(f"\nResults saved to: {results_file}")
    return results


if __name__ == "__main__":
    import sys
    shelf_directory = sys.argv[1] if len(sys.argv) > 1 else "../divided_world_1"
    results = run_experiment(shelf_directory)