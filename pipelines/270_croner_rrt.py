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
    """
    current_pos = (current_node.x, current_node.y)
    
    # First check: Can we see the target directly?
    if is_target_visible(current_pos, goal, grid):
        return goal[0] - 0.5, goal[1] - 0.5  # Convert to grid coordinates
    
    # If no edge of coverage, fall back to random sampling
    if not edge_of_coverage:
        return random.randint(0, grid.shape[1] - 1), random.randint(0, grid.shape[0] - 1)
    
    # Get visible 270-degree corners
    visible_corners = get_visible_corners(current_pos, corner_list, grid)
    
    if not visible_corners:
        # No corners visible, sample from edge of coverage
        return random.choice(list(edge_of_coverage))
    
    # Select the best corner based on edge coverage intersection
    best_corner, intersection_size = select_best_corner(visible_corners, edge_of_coverage, grid, walkable_cells)
    
    if best_corner and intersection_size > 0:
        return best_corner
    else:
        # Fallback to random corner if no good intersection found
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
                    return nodes, len(nodes)

    return None, 0

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
    """Main experimental pipeline for 270 Corner RRT"""
    print("=" * 60)
    print("270 CORNER RRT EXPERIMENTAL PIPELINE")
    print("=" * 60)
    
    # Create results directory
    results_dir = Path("corner_rrt_results")
    results_dir.mkdir(exist_ok=True)
    
    # Load existing shelf worlds
    print(f"Loading shelf worlds from directory: {shelf_dir}")
    worlds, shelf_names = load_shelf_worlds(shelf_dir, num_worlds=60)
    
    if len(worlds) == 0:
        print("ERROR: No worlds loaded! Please check the directory path and file format.")
        return []
    
    results = []
    total_runs = 0
    successful_runs = 0
    
    # For each world
    for world_id, (grid, shelf_name) in enumerate(zip(worlds, shelf_names)):
        print(f"\nShelf {world_id + 1}/{len(worlds)}: {shelf_name}")
        print(f"  Grid size: {grid.shape}")
        
        # Check if this is the dungeon_4251 world
        is_dungeon_4251 = "dungeon_4251" in shelf_name
        if is_dungeon_4251:
            print(f"  *** FOUND TARGET WORLD: {shelf_name} ***")
        
        # Create walkable cells set for this world
        walkable_cells = create_walkable_cells(grid)
        print(f"  Found {len(walkable_cells)} walkable cells")
        
        # Find 270-degree corners for this world
        corners_270 = find_270_corners(grid)
        print(f"  Found {len(corners_270)} 270-degree corners")
        
        # Generate 5 start-goal pairs
        pairs = generate_start_goal_pairs(grid, num_pairs=5)
        print(f"  Generated {len(pairs)} start-goal pairs")
        
        # Test each pair
        for pair_id, (start, goal) in enumerate(pairs):
            print(f"  Pair {pair_id + 1}: Start{start} -> Goal{goal}")
            
            # Run with 5 different seeds
            for seed_id in range(5):
                seed_value = 100 + seed_id  # Seeds: 100, 101, 102, 103, 104
                random.seed(seed_value)
                np.random.seed(seed_value)
                
                print(f"    Seed {seed_id + 1} (value: {seed_value})")
                
                # Run 270 Corner RRT
                start_time = time.time()
                nodes, node_count = enhanced_rrt(grid, start, goal, corners_270, walkable_cells, step_size=2, max_iter=5000)
                end_time = time.time()
            
                execution_time = end_time - start_time
                total_runs += 1
                
                if nodes:
                    successful_runs += 1
                    path_len = path_length(nodes[-1])
                    success = True
                    print(f"    SUCCESS: {node_count} nodes, {path_len} path length, {execution_time:.2f}s")
                else:
                    path_len = 0
                    success = False
                    print(f"    FAILED: No path found, {execution_time:.2f}s")
                
                # Save PNG visualization if this is dungeon_4251
                if is_dungeon_4251:
                    viz_filename = f"dungeon_4251_pair_{pair_id + 1}_{'SUCCESS' if success else 'FAILED'}.png"
                    print(f"    *** SAVING VISUALIZATION: {viz_filename} ***")
                    visualize_rrt_result(grid, nodes if nodes else [], start, goal, corners_270, viz_filename)
                
                # Store results
                result = {
                    'world_id': world_id,
                    'shelf_name': shelf_name,
                    'pair_id': pair_id,
                    'start': start,
                    'goal': goal,
                    'success': success,
                    'nodes_explored': node_count,
                    'path_length': path_len,
                    'execution_time': execution_time,
                    'algorithm': '270_corner_rrt',
                    'corners_found': len(corners_270)
                }
                results.append(result)
    
    # Save results
    results_file = results_dir / "corner_rrt_results.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Print summary statistics
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
            avg_path_length = np.mean([r['path_length'] for r in successful_results])
            avg_time = np.mean([r['execution_time'] for r in successful_results])
            avg_corners = np.mean([r['corners_found'] for r in results])
            
            print(f"Average nodes explored: {avg_nodes:.1f}")
            print(f"Average path length: {avg_path_length:.1f}")
            print(f"Average execution time: {avg_time:.2f}s")
            print(f"Average corners found per world: {avg_corners:.1f}")
    else:
        print("No runs completed - check if worlds were loaded correctly!")
    
    print(f"\nResults saved to: {results_file}")
    return results

def create_dungeon_4251_visualization(shelf_dir=".", save_path="dungeon_4251_world.png"):
    """
    Create a standalone visualization of the dungeon_4251 world without RRT
    """
    print("Creating dungeon_4251 world visualization...")
    
    # Load worlds and find dungeon_4251
    worlds, shelf_names = load_shelf_worlds(shelf_dir, num_worlds=100)
    
    dungeon_4251_grid = None
    dungeon_4251_name = None
    
    for grid, shelf_name in zip(worlds, shelf_names):
        if "dungeon_4251" in shelf_name:
            dungeon_4251_grid = grid
            dungeon_4251_name = shelf_name
            print(f"Found dungeon_4251: {shelf_name}")
            break
    
    if dungeon_4251_grid is None:
        print("ERROR: dungeon_4251 not found in the loaded worlds!")
        return
    
    # Create walkable cells and find corners
    walkable_cells = create_walkable_cells(dungeon_4251_grid)
    corners_270 = find_270_corners(dungeon_4251_grid)
    
    fig, ax = plt.subplots(1, 1, figsize=(15, 15))
    
    # Create RGB visualization of the dungeon world
    height, width = dungeon_4251_grid.shape
    rgb_grid = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Set colors: obstacles=black, free space=white
    free_mask = dungeon_4251_grid > 128  # Free space
    obstacle_mask = dungeon_4251_grid <= 128  # Obstacles
    
    rgb_grid[free_mask] = [255, 255, 255]  # Pure white for free space
    rgb_grid[obstacle_mask] = [0, 0, 0]    # Black for obstacles
    
    # Show the dungeon world
    ax.imshow(rgb_grid, origin='lower', extent=[0, width, 0, height])
    
    # Plot 270-degree corners as blue squares
    if corners_270:
        corner_x = [c[0] + 0.5 for c in corners_270]  # Center in cell
        corner_y = [c[1] + 0.5 for c in corners_270]
        ax.scatter(corner_x, corner_y, c='blue', s=100, marker='s', 
                  label=f'270° Corners ({len(corners_270)})', alpha=0.8, 
                  edgecolors='navy', linewidths=1, zorder=5)
    
    # Calculate statistics
    free_cells = len(walkable_cells)
    total_cells = width * height
    free_ratio = free_cells / total_cells
    
    # Formatting
    ax.set_xlim(0, width)
    ax.set_ylim(0, height)
    ax.set_aspect('equal')
    
    # Add legend if there are corners
    if corners_270:
        ax.legend(loc='upper right', fontsize=12, framealpha=0.9)
    
    title = f'Dungeon 4251 World Layout\n'
    title += f'Size: {width}×{height}, Free Space: {free_ratio:.1%}, '
    title += f'270° Corners: {len(corners_270)}'
    
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('X coordinate', fontsize=12)
    ax.set_ylabel('Y coordinate', fontsize=12)
    
    # Remove ticks for cleaner look
    ax.set_xticks([])
    ax.set_yticks([])
    
    # Save the figure with high resolution
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white', 
                edgecolor='none')
    plt.close()
    print(f"Dungeon 4251 visualization saved to: {save_path}")
    
    # Save detailed analysis
    analysis_path = save_path.replace('.png', '_analysis.txt')
    with open(analysis_path, 'w') as f:
        f.write(f"Dungeon 4251 World Analysis\n")
        f.write(f"==========================\n")
        f.write(f"Source: {dungeon_4251_name}\n")
        f.write(f"Grid size: {width} × {height}\n")
        f.write(f"Total cells: {total_cells}\n")
        f.write(f"Free cells: {free_cells} ({free_ratio:.1%})\n")
        f.write(f"Obstacle cells: {total_cells - free_cells} ({1-free_ratio:.1%})\n")
        f.write(f"270° corners found: {len(corners_270)}\n")
        if corners_270:
            f.write(f"Corner positions: {corners_270}\n")
        f.write(f"\nWorld Characteristics\n")
        f.write(f"===================\n")
        f.write(f"This appears to be a dungeon-style environment with:\n")
        f.write(f"- Corridors and rooms (white areas)\n")
        f.write(f"- Walls and obstacles (black areas)\n")
        f.write(f"- Strategic 270° corners for navigation\n")
    
    print(f"Analysis saved to: {analysis_path}")
    return dungeon_4251_grid, corners_270

def visualize_rrt_result(grid, nodes, start, goal, corner_list, save_path="rrt_result.png"):
    """
    Visualize the RRT result showing the actual world, corners, tree, and path
    """
    fig, ax = plt.subplots(1, 1, figsize=(15, 15))
    
    # Create RGB visualization of the actual world
    # Black = obstacles (0), White = free space (255)
    height, width = grid.shape
    rgb_grid = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Set colors: obstacles=black, free space=light gray
    free_mask = grid > 128  # Free space
    obstacle_mask = grid <= 128  # Obstacles
    
    rgb_grid[free_mask] = [240, 240, 240]  # Light gray for free space
    rgb_grid[obstacle_mask] = [0, 0, 0]    # Black for obstacles
    
    # Show the actual world
    ax.imshow(rgb_grid, origin='lower', extent=[0, width, 0, height])
    
    # Plot 270-degree corners as blue squares
    if corner_list:
        corner_x = [c[0] + 0.5 for c in corner_list]  # Center in cell
        corner_y = [c[1] + 0.5 for c in corner_list]
        ax.scatter(corner_x, corner_y, c='blue', s=150, marker='s', 
                  label=f'270° Corners ({len(corner_list)})', alpha=0.9, 
                  edgecolors='white', linewidths=2, zorder=5)
        
        # Print corner positions
        print(f"270° Corners detected at: {corner_list}")
    
    # Plot RRT tree in green
    if nodes and len(nodes) > 1:
        tree_edges_x = []
        tree_edges_y = []
        
        for i, node in enumerate(nodes[1:], 1):  # Skip start node
            if node.parent:
                tree_edges_x.extend([node.parent.x, node.x, None])
                tree_edges_y.extend([node.parent.y, node.y, None])
        
        # Draw all tree edges at once
        ax.plot(tree_edges_x, tree_edges_y, 'g-', alpha=0.4, linewidth=1, 
               label=f'RRT Tree ({len(nodes)} nodes)', zorder=2)
        
        # Extract and highlight the final path in red
        if nodes and nodes[-1].parent:  # If goal was reached
            path_nodes = []
            current = nodes[-1]
            while current:
                path_nodes.append(current)
                current = current.parent
            path_nodes.reverse()
            
            # Draw final path in red with thick line
            path_x = [node.x for node in path_nodes]
            path_y = [node.y for node in path_nodes]
            ax.plot(path_x, path_y, 'r-', linewidth=4, alpha=0.9, 
                   label=f'Final Path ({len(path_nodes)} nodes)', zorder=4)
            
            print(f"Path found with {len(path_nodes)} waypoints")
    
    # Plot start as large green circle
    ax.scatter(start[0] + 0.5, start[1] + 0.5, c='lime', s=400, marker='o', 
              label='Start', edgecolors='darkgreen', linewidths=3, zorder=6)
    
    # Plot goal as large red star
    ax.scatter(goal[0] + 0.5, goal[1] + 0.5, c='red', s=500, marker='*', 
              label='Goal', edgecolors='darkred', linewidths=3, zorder=6)
    
    # Formatting
    ax.set_xlim(0, width)
    ax.set_ylim(0, height)
    ax.set_aspect('equal')
    ax.legend(loc='upper left', bbox_to_anchor=(0, 1), fontsize=12)
    
    # Calculate some statistics
    free_cells = np.sum(grid > 128)
    total_cells = grid.size
    free_ratio = free_cells / total_cells
    
    title = f'270° Corner RRT on Actual World\n'
    title += f'Start({start[0]}, {start[1]}) → Goal({goal[0]}, {goal[1]})\n'
    title += f'World: {width}×{height}, Free Space: {free_ratio:.1%}, '
    title += f'Corners: {len(corner_list) if corner_list else 0}'
    
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    ax.set_xlabel('X coordinate', fontsize=12)
    ax.set_ylabel('Y coordinate', fontsize=12)
    
    # Add a subtle grid
    ax.grid(True, alpha=0.2, linewidth=0.5)
    
    # Save the figure with high resolution
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"High-resolution visualization saved to: {save_path}")
    
    # Also save world analysis
    analysis_path = save_path.replace('.png', '_analysis.txt')
    with open(analysis_path, 'w') as f:
        f.write(f"World Analysis\n")
        f.write(f"=============\n")
        f.write(f"Grid size: {width} × {height}\n")
        f.write(f"Total cells: {total_cells}\n")
        f.write(f"Free cells: {free_cells} ({free_ratio:.1%})\n")
        f.write(f"Obstacle cells: {total_cells - free_cells} ({1-free_ratio:.1%})\n")
        f.write(f"270° corners found: {len(corner_list) if corner_list else 0}\n")
        if corner_list:
            f.write(f"Corner positions: {corner_list}\n")
        f.write(f"\nPath Planning Results\n")
        f.write(f"====================\n")
        f.write(f"Start position: {start}\n")
        f.write(f"Goal position: {goal}\n")
        if nodes:
            path_length_val = path_length(nodes[-1])
            f.write(f"Nodes explored: {len(nodes)}\n")
            f.write(f"Path length: {path_length_val:.2f}\n")
            f.write(f"Success: Yes\n")
        else:
            f.write(f"Success: No\n")
    
    print(f"Analysis saved to: {analysis_path}")

def run_single_visualization_example(shelf_dir=".", shelf_index=0, pair_index=0):
    """
    Run a single example and save visualization
    """
    print("=" * 60)
    print("270 CORNER RRT - SINGLE VISUALIZATION EXAMPLE")
    print("=" * 60)
    
    # Load worlds
    worlds, shelf_names = load_shelf_worlds(shelf_dir, num_worlds=min(10, shelf_index + 1))
    
    if len(worlds) <= shelf_index:
        print(f"ERROR: Not enough worlds loaded. Requested index {shelf_index}, but only {len(worlds)} available.")
        return
    
    # Select the specific world
    grid = worlds[shelf_index]
    shelf_name = shelf_names[shelf_index]
    
    print(f"Using Shelf {shelf_index + 1}: {shelf_name}")
    print(f"Grid size: {grid.shape}")
    
    # Create walkable cells and find corners
    walkable_cells = create_walkable_cells(grid)
    corners_270 = find_270_corners(grid)
    print(f"Found {len(walkable_cells)} walkable cells")
    print(f"Found {len(corners_270)} 270-degree corners")
    
    # Generate start-goal pairs
    pairs = generate_start_goal_pairs(grid, num_pairs=max(5, pair_index + 1))
    
    if len(pairs) <= pair_index:
        print(f"ERROR: Not enough pairs generated. Requested index {pair_index}, but only {len(pairs)} available.")
        return
    
    start, goal = pairs[pair_index]
    print(f"Using Pair {pair_index + 1}: Start{start} -> Goal{goal}")
    
    # Run 270 Corner RRT
    print("Running 270 Corner RRT...")
    start_time = time.time()
    nodes, node_count = enhanced_rrt(grid, start, goal, corners_270, walkable_cells, 
                                   step_size=2, max_iter=5000)
    end_time = time.time()
    
    execution_time = end_time - start_time
    
    if nodes:
        path_len = path_length(nodes[-1])
        print(f"SUCCESS: {node_count} nodes, {path_len:.2f} path length, {execution_time:.2f}s")
        
        # Create visualization
        viz_filename = f"corner_rrt_shelf_{shelf_index}_pair_{pair_index}.png"
        visualize_rrt_result(grid, nodes, start, goal, corners_270, viz_filename)
        
    else:
        print(f"FAILED: No path found, {execution_time:.2f}s")
        # Still create visualization to show the environment
        viz_filename = f"corner_rrt_shelf_{shelf_index}_pair_{pair_index}_FAILED.png"
        visualize_rrt_result(grid, nodes, start, goal, corners_270, viz_filename)

def debug_failed_cases(shelf_dir="."):
    """
    Debug the specific failed cases
    """
    print("=" * 60)
    print("DEBUGGING FAILED CASES")
    print("=" * 60)
    
    # Load the specific worlds that failed
    failed_cases = [
        {"shelf_pattern": "16000-16999_dungeon_2210_4", "pairs": [(18, 40, 55, 8), (13, 29, 82, 66)]},
        {"shelf_pattern": "16000-16999_dungeon_2691_3", "pairs": [(60, 11, 25, 68)]}
    ]
    
    worlds, shelf_names = load_shelf_worlds(shelf_dir, num_worlds=60)
    
    for case in failed_cases:
        # Find the matching shelf
        matching_indices = [i for i, name in enumerate(shelf_names) 
                          if case["shelf_pattern"] in name]
        
        if not matching_indices:
            print(f"Could not find shelf matching pattern: {case['shelf_pattern']}")
            continue
            
        shelf_idx = matching_indices[0]
        grid = worlds[shelf_idx]
        shelf_name = shelf_names[shelf_idx]
        
        print(f"\nAnalyzing failed shelf: {shelf_name}")
        print(f"Grid size: {grid.shape}")
        
        # Analyze the grid
        walkable_cells = create_walkable_cells(grid)
        corners_270 = find_270_corners(grid)
        
        free_ratio = len(walkable_cells) / (grid.shape[0] * grid.shape[1])
        print(f"Walkable cells: {len(walkable_cells)} ({free_ratio:.1%})")
        print(f"270-degree corners: {len(corners_270)}")
        
        # Test each failed pair
        for start_x, start_y, goal_x, goal_y in case["pairs"]:
            start = (start_x, start_y)
            goal = (goal_x, goal_y)
            print(f"\nTesting failed pair: Start{start} -> Goal{goal}")
            
            # Check if start and goal are in walkable cells
            if start not in walkable_cells:
                print(f"  ERROR: Start position {start} is not walkable!")
            if goal not in walkable_cells:
                print(f"  ERROR: Goal position {goal} is not walkable!")
            
            # Check direct distance
            distance = math.sqrt((goal_x - start_x)**2 + (goal_y - start_y)**2)
            print(f"  Direct distance: {distance:.2f}")
            
            # Try to find path with more iterations
            print("  Attempting with increased iterations...")
            start_time = time.time()
            nodes, node_count = enhanced_rrt(grid, start, goal, corners_270, walkable_cells, 
                                           step_size=1, max_iter=10000)  # Smaller steps, more iterations
            end_time = time.time()
            
            if nodes:
                path_len = path_length(nodes[-1])
                print(f"  SUCCESS with increased iterations: {node_count} nodes, {path_len:.2f} path length, {end_time - start_time:.2f}s")
            else:
                print(f"  Still FAILED even with increased iterations ({end_time - start_time:.2f}s)")
            
            # Create visualization of this failed case
            viz_filename = f"debug_failed_{shelf_name}_start_{start_x}_{start_y}_goal_{goal_x}_{goal_y}.png"
            visualize_rrt_result(grid, nodes if nodes else [], start, goal, corners_270, viz_filename)


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "create_dungeon":
        # Create standalone dungeon_4251 visualization
        shelf_directory = sys.argv[2] if len(sys.argv) > 2 else "../divided_world_1"
        create_dungeon_4251_visualization(shelf_directory, "dungeon_4251_world.png")
    elif len(sys.argv) > 1 and sys.argv[1] == "debug":
        # Debug failed cases
        shelf_directory = sys.argv[2] if len(sys.argv) > 2 else "../divided_world_1"
        debug_failed_cases(shelf_directory)
    elif len(sys.argv) >= 3:
        # Run single visualization
        shelf_directory = sys.argv[1] if sys.argv[1] != "." else "../divided_world_1"
        shelf_index = int(sys.argv[2]) if sys.argv[2].isdigit() else 0
        pair_index = int(sys.argv[3]) if len(sys.argv) > 3 and sys.argv[3].isdigit() else 0
        run_single_visualization_example(shelf_directory, shelf_index, pair_index)
    else:
        # Default: run full experiment
        shelf_directory = sys.argv[1] if len(sys.argv) > 1 else "../divided_world_1"
        results = run_experiment(shelf_directory)