import numpy as np
import matplotlib.pyplot as plt
import math
import random
import heapq
import json
import time
import os
from pathlib import Path
import networkx as nx

SEED = 18  # or parameterize this
random.seed(SEED)
np.random.seed(SEED)


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


def build_free_graph(grid):
    """Build a NetworkX graph of all free cells in the grid."""
    G = nx.Graph()
    height, width = grid.shape
    for y in range(height):
        for x in range(width):
            if grid[y, x] != 0:  # free cell
                G.add_node((x, y))
                # connect to neighbors
                for dx, dy in [(-1,0), (1,0), (0,-1), (0,1),
               (-1,-1), (-1,1), (1,-1), (1,1)]:
                    nx_, ny_ = x+dx, y+dy
                    if 0 <= nx_ < width and 0 <= ny_ < height and grid[ny_, nx_] != 0:
                        G.add_edge((x, y), (nx_, ny_), weight=1)
    return G


def nearest_node(nodes, x, y):
    """Find nearest node in the tree to (x,y) using Euclidean distance."""
    # Keep this simple (no need for NX here unless you want global shortest path)
    return min(nodes, key=lambda node: (node.x - x)**2 + (node.y - y)**2)

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


def collision(x1, y1, x2, y2, free_graph):
    return not nx.has_path(free_graph, (x1, y1), (x2, y2))

    

def extract_path(goal_node):
    """Backtrack final path from goal_node to start."""
    path = []
    current = goal_node
    while current:
        path.append((current.x, current.y))
        current = current.parent
    return list(reversed(path))


def rrt(grid, start, goal, step_size=10, max_iter=1000):
    free_graph = build_free_graph(grid)
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
            if (new_x, new_y) not in visited and not collision(nearest.x, nearest.y, new_x, new_y, free_graph):
                new_node = Node(new_x, new_y)
                new_node.parent = nearest
                nodes.append(new_node)
                visited.add((new_x, new_y))

                if abs(new_x - goal[0]) + abs(new_y - goal[1]) <= 1:
                    goal_node.parent = new_node
                    nodes.append(goal_node)
                    
                    # Calculate path efficiency
                    path_nodes = set()
                    current = goal_node  # Start from goal
                    while current is not None:
                        if hasattr(current, 'x') and hasattr(current, 'y'):
                            path_nodes.add((int(current.x), int(current.y)))
                        current = current.parent
                    
                    nodes_on_final_path = len(path_nodes)
                    return nodes, len(nodes), nodes_on_final_path

    return None, 0, 0
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
        
        # Set seed once per world to ensure consistent start-goal pairs across all RRT algorithms
        world_seed = 100 + world_id  # Each world gets its own seed
        random.seed(world_seed)
        np.random.seed(world_seed)
        print(f"  World seed: {world_seed}")
        
        # Generate 5 start-goal pairs (these will be the same for all RRT algorithms)
        pairs = generate_start_goal_pairs(grid, num_pairs=5)
        print(f"  Generated {len(pairs)} start-goal pairs")
        
        # Test each pair (FIXED INDENTATION - now properly inside world loop)
        for pair_id, (start, goal) in enumerate(pairs):
            print(f"  Pair {pair_id + 1}: Start{start} -> Goal{goal}")
            
            # Reset to world seed before each RRT run for algorithm consistency
            random.seed(world_seed)
            np.random.seed(world_seed)
            
            print(f"    Running pair {pair_id + 1} with world seed {world_seed}")
            
            # Run Standard RRT
            start_time = time.time()
            nodes, node_count, path_node_count = rrt(grid, start, goal, step_size=2, max_iter=5000)
            end_time = time.time()
            
            execution_time = end_time - start_time
            total_runs += 1
            
            if nodes:
                successful_runs += 1
                path_len = path_length(nodes[-1])
                success = True
                
                # Calculate efficiency
                path_efficiency = path_node_count / node_count if node_count > 0 else 0
                
                print(f"      SUCCESS: {node_count} nodes explored, {path_node_count} on final path ({path_efficiency:.1%} efficiency), {path_len:.1f} path length, {execution_time:.2f}s")
            else:
                path_len = 0
                path_node_count = 0
                success = False
                print(f"      FAILED: No path found, {execution_time:.2f}s")
            
            # Store results
            result = {
                'world_id': world_id,
                'shelf_name': shelf_name,
                'pair_id': pair_id,
                'world_seed': world_seed,
                'start': start,
                'goal': goal,
                'success': success,
                'nodes_explored': node_count,
                'nodes_on_final_path': path_node_count,
                'path_efficiency': path_node_count / node_count if node_count > 0 else 0,
                'path_length': path_len,
                'execution_time': execution_time,
                'algorithm': 'standard_rrt'
            }
            results.append(result)
        
        # Per-world summary
        world_results = [r for r in results if r['world_id'] == world_id]
        world_successful = sum(1 for r in world_results if r['success'])
        world_total = len(world_results)
        
        print(f"\n  --- World {world_id + 1} Summary ---")
        print(f"  Total runs: {world_total}")
        print(f"  Successful runs: {world_successful}")
        
        if world_total > 0:
            print(f"  Success rate: {world_successful/world_total*100:.1f}%")
            
            if world_successful > 0:
                successful_world_results = [r for r in world_results if r['success']]
                avg_nodes = np.mean([r['nodes_explored'] for r in successful_world_results])
                avg_path_nodes = np.mean([r['nodes_on_final_path'] for r in successful_world_results])
                avg_efficiency = np.mean([r['path_efficiency'] for r in successful_world_results])
                avg_path_length = np.mean([r['path_length'] for r in successful_world_results])
                avg_time = np.mean([r['execution_time'] for r in successful_world_results])
                
                print(f"  Average nodes explored: {avg_nodes:.1f}")
                print(f"  Average nodes on final path: {avg_path_nodes:.1f}")
                print(f"  Average path efficiency: {avg_efficiency:.1%}")
                print(f"  Average path length: {avg_path_length:.1f}")
                print(f"  Average execution time: {avg_time:.2f}s")
    
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
            avg_path_nodes = np.mean([r['nodes_on_final_path'] for r in successful_results])
            avg_efficiency = np.mean([r['path_efficiency'] for r in successful_results])
            avg_path_length = np.mean([r['path_length'] for r in successful_results])
            avg_time = np.mean([r['execution_time'] for r in successful_results])
            
            print(f"Average nodes explored: {avg_nodes:.1f}")
            print(f"Average nodes on final path: {avg_path_nodes:.1f}")
            print(f"Average path efficiency: {avg_efficiency:.1%}")
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