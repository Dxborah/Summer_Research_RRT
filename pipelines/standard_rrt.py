import numpy as np
import matplotlib.pyplot as plt
import math
import random
import heapq
import json
import time
import os
from pathlib import Path

class Node:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.parent = None

def distance_angle(x1, y1, x2, y2):
    distance = math.sqrt(((x1 - x2)**2) + ((y1 - y2)**2))
    dy = y2 - y1
    dx = x2 - x1
    angle = math.atan2(dy, dx)
    return (distance, angle)

def nearest_node(node_list, x, y):
    closest_node = None
    min_distance = float('inf')
    for node in node_list:
        distance = math.sqrt(((node.x - x)**2) + ((node.y - y)**2))
        if distance < min_distance:
            min_distance = distance
            closest_node = node
    return closest_node

def random_point(grid_size, goal, goal_sample_rate=0.1):
    if random.random() < goal_sample_rate:
        return goal
    else:
        return (random.randint(0, grid_size - 1), random.randint(0, grid_size - 1))

def steer(from_node, to_x, to_y, grid, step_size=10):
    dist, angle = distance_angle(from_node.x, from_node.y, to_x, to_y)
    dist = min(dist, step_size)
    new_x = int(from_node.x + dist * math.cos(angle))
    new_y = int(from_node.y + dist * math.sin(angle))

    if 0 <= new_x < grid.shape[1] and 0 <= new_y < grid.shape[0] and grid[new_y, new_x] != 0:
        return (new_x, new_y)
    else:
        return None

def is_moving_towards_goal(current, next_node, goal):
    dx1 = goal[0] - current[0]
    dy1 = goal[1] - current[1]
    dx2 = next_node[0] - current[0]
    dy2 = next_node[1] - current[1]
    return dx1 * dx2 + dy1 * dy2 > 0

def interpolate_path(x1, y1, x2, y2, grid):
    height, width = grid.shape
    start = (x1, y1)
    goal = (x2, y2)

    if grid[y1, x1] == 0 or grid[y2, x2] == 0:
        return []

    visited = set()
    parent = {}
    cost = {start: 0}
    heap = [(0, start)]

    directions = [(-1, 0), (1, 0), (0, -1), (0, 1), 
                  (-1, -1), (-1, 1), (1, -1), (1, 1)]

    while heap:
        curr_cost, current = heapq.heappop(heap)
        if current in visited:
            continue
        visited.add(current)

        if current == goal:
            break

        for dx, dy in directions:
            nx, ny = current[0] + dx, current[1] + dy
            if 0 <= nx < width and 0 <= ny < height and grid[ny, nx] != 0:
                next_node = (nx, ny)
                new_cost = cost[current] + 1
                if is_moving_towards_goal(current, next_node, goal):
                    if next_node not in cost or new_cost < cost[next_node]:
                        cost[next_node] = new_cost
                        parent[next_node] = current
                        heapq.heappush(heap, (new_cost, next_node))

    if goal not in parent:
        return []
    
    path = []
    node = goal
    while node != start:
        path.append(node)
        node = parent[node]
    path.append(start)
    path.reverse()
    return path

def collision(x1, y1, x2, y2, grid):
    path = interpolate_path(x1, y1, x2, y2, grid)
    return len(path) == 0

def rrt(grid, start, goal, step_size=10, max_iter=1000):
    visited = set()
    start_node = Node(start[0], start[1])  
    goal_node = Node(goal[0], goal[1])  
    visited.add((start_node.x, start_node.y))  
    nodes = [start_node]  

    for _ in range(max_iter):
        rand_x, rand_y = random_point(grid.shape[0], goal)
        nearest = nearest_node(nodes, rand_x, rand_y)
        steered = steer(nearest, rand_x, rand_y, grid, step_size=step_size)

        if steered:
            new_x, new_y = steered
            if (new_x, new_y) not in visited and not collision(nearest.x, nearest.y, new_x, new_y, grid):
                new_node = Node(new_x, new_y)
                new_node.parent = nearest
                nodes.append(new_node)
                visited.add((new_x, new_y))

                if abs(new_x - goal[0]) + abs(new_y - goal[1]) <= 1:
                    goal_node.parent = new_node
                    nodes.append(goal_node)
                    return nodes, len(nodes)

    return None, 0

def path_length(goal_node):
    if goal_node is None:
        return 0
    path = []
    node = goal_node
    while node is not None:
        path.append((node.x, node.y))
        node = node.parent
    return len(path)

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
    """Main experimental pipeline for Standard RRT"""
    print("=" * 60)
    print("STANDARD RRT EXPERIMENTAL PIPELINE")
    print("=" * 60)
    
    # Create results directory
    results_dir = Path("standard_rrt_results")
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
        
        # Generate 5 start-goal pairs
        pairs = generate_start_goal_pairs(grid, num_pairs=5)
        print(f"  Generated {len(pairs)} start-goal pairs")
        
        # Test each pair
        for pair_id, (start, goal) in enumerate(pairs):
            print(f"  Pair {pair_id + 1}: Start{start} -> Goal{goal}")
            
            # Run Standard RRT
            start_time = time.time()
            nodes, node_count = rrt(grid, start, goal, step_size=2, max_iter=5000)
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
                'algorithm': 'standard_rrt'
            }
            results.append(result)
    
    # Save results
    results_file = results_dir / "standard_rrt_results.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Print summary statistics
    print("\n" + "=" * 60)
    print("STANDARD RRT - FINAL RESULTS SUMMARY")
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
            
            print(f"Average nodes explored: {avg_nodes:.1f}")
            print(f"Average path length: {avg_path_length:.1f}")
            print(f"Average execution time: {avg_time:.2f}s")
    else:
        print("No runs completed - check if worlds were loaded correctly!")
    
    print(f"\nResults saved to: {results_file}")
    return results

if __name__ == "__main__":
    import sys
    shelf_directory = sys.argv[1] if len(sys.argv) > 1 else "."
    results = run_experiment("../divided_world_1")