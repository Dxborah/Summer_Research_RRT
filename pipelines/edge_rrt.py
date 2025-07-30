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
        self.x = x + 0.5
        self.y = y + 0.5
        self.parent = None

# Global grids for visibility and pheromones
visible_grid = None
pheromone_grid = None
walkable_cells = None
grid = None

def grid_value(x, y, grid):
    ix = int(math.floor(x))
    iy = int(math.floor(y))
    if 0 <= iy < grid.shape[0] and 0 <= ix < grid.shape[1]:
        return grid[iy, ix]
    return 0

def distance_angle(x1, y1, x2, y2):
    distance = math.hypot(x2 - x1, y2 - y1)
    angle = math.atan2(y2 - y1, x2 - x1)
    return (distance, angle)

def nearest_node(node_list, x, y):
    return min(node_list, key=lambda node: math.hypot(node.x - x, node.y - y))

def compute_edge_of_coverage(grid, visible_grid, walkable_cells):
    height, width = grid.shape
    seen_white = set()
    edge_candidates = set()
    unseen_white = set()

    for (x, y) in walkable_cells:
        if visible_grid[y, x] == 1:
            seen_white.add((x, y))
        else:
            unseen_white.add((x, y))

    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    for (x, y) in seen_white:
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if (nx, ny) in walkable_cells and visible_grid[ny, nx] == 0:
                edge_candidates.add((nx, ny))

    edge_of_coverage = edge_candidates & unseen_white
    return edge_of_coverage

def ant_random_point(goal, current_node, edge_of_coverage, pheromone_grid, grid_size, goal_sample_rate=0.05):
    global walkable_cells
    
    if random.random() < goal_sample_rate:
        return goal

    if not edge_of_coverage:
        edge_of_coverage = compute_edge_of_coverage(grid, visible_grid, walkable_cells)
        if not edge_of_coverage:
            return random.randint(0, grid_size - 1), random.randint(0, grid_size - 1)

    edge_list = list(edge_of_coverage)
    weights = [pheromone_grid[y, x] + 1e-6 for (x, y) in edge_list]
    total = sum(weights)

    if total <= 0 or not np.isfinite(total):
        return random.choice(edge_list)

    probs = np.array(weights) / total
    sampled_index = np.random.choice(len(edge_list), p=probs)
    return edge_list[sampled_index]

def steer(from_node, to_x, to_y, grid, step_size):
    distance, angle = distance_angle(from_node.x, from_node.y, to_x + 0.5, to_y + 0.5)
    step = min(step_size, distance)

    for i in np.linspace(1, step, int(step) + 1):
        new_x = from_node.x + i * math.cos(angle)
        new_y = from_node.y + i * math.sin(angle)
        val = grid_value(new_x, new_y, grid)
        if val == 0:
            return None
    return (from_node.x + step * math.cos(angle), from_node.y + step * math.sin(angle))

def update_visibility(x, y, grid, visible_grid, max_distance=None):
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

def interpolate_path(x1, y1, x2, y2, grid):
    x1, y1 = int(math.floor(x1)), int(math.floor(y1))
    x2, y2 = int(math.floor(x2)), int(math.floor(y2))

    if grid[y1, x1] == 0 or grid[y2, x2] == 0:
        return []

    height, width = grid.shape
    start = (x1, y1)
    goal = (x2, y2)

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
    return len(interpolate_path(x1, y1, x2, y2, grid)) == 0

def path_length(goal_node):
    if goal_node is None:
        return 0
    path = []
    node = goal_node
    while node is not None:
        path.append((node.x, node.y))
        node = node.parent
    return len(path)

def create_walkable_cells(grid):
    """Create set of walkable cells from grid"""
    walkable_cells = set()
    height, width = grid.shape
    for y in range(height):
        for x in range(width):
            if grid[y, x] == 255:  # White cells are walkable
                walkable_cells.add((x, y))
    return walkable_cells

def rrt_edge_sampling(grid_param, start, goal, walkable_cells_param, step_size=10, max_iter=1000):
    global pheromone_grid, visible_grid, walkable_cells, grid
    
    # Initialize global grids and walkable cells
    grid = grid_param
    visible_grid = np.zeros_like(grid, dtype=np.uint8)
    pheromone_grid = np.zeros_like(grid, dtype=np.float32)
    walkable_cells = walkable_cells_param

    start_node = Node(*start)
    goal_node = Node(*goal)

    update_visibility(start_node.x, start_node.y, grid, visible_grid)
    nodes = [start_node]

    visited = set([(int(start_node.x), int(start_node.y))])

    for _ in range(max_iter):
        edge_of_coverage = compute_edge_of_coverage(grid, visible_grid, walkable_cells)

        rand_x, rand_y = ant_random_point(goal, nodes[-1], edge_of_coverage, pheromone_grid, grid.shape[0])
                
        nearest = nearest_node(nodes, rand_x + 0.5, rand_y + 0.5)
        steered = steer(nearest, rand_x, rand_y, grid, step_size)

        if steered:
            new_x, new_y = steered
            ix, iy = int(math.floor(new_x)), int(math.floor(new_y))

            if (ix, iy) not in visited and not collision(nearest.x, nearest.y, new_x, new_y, grid):
                new_node = Node(ix, iy)
                new_node.x, new_node.y = new_x, new_y
                new_node.parent = nearest
                nodes.append(new_node)
                visited.add((ix, iy))

                update_visibility(new_x, new_y, grid, visible_grid)
                pheromone_grid *= 0.998

                if abs(new_x - goal_node.x) + abs(new_y - goal_node.y) <= 3:
                    goal_node.parent = new_node
                    nodes.append(goal_node)
                    return nodes, len(nodes)

    return None, 0

def load_shelf_worlds(shelf_dir=".", num_worlds=60):
    """Load existing shelf worlds from .dat/.txt/.csv files, searching recursively"""
    import struct
    import glob
    
    worlds = []
    shelf_files = []

    print(f"Searching for grid files recursively in: {shelf_dir}")
    shelf_path = Path(shelf_dir)
    
    if not shelf_path.exists():
        print(f"Directory {shelf_dir} does not exist!")
        return [], []

    # Collect all .dat, .txt, and .csv files
    all_files = list(shelf_path.rglob("*"))
    grid_files = [f for f in all_files if f.suffix in [".dat", ".txt", ".csv"]]
    grid_files = [str(f) for f in grid_files]
    grid_files.sort()

    if not grid_files:
        print("No grid files found!")
        return [], []

    if len(grid_files) < num_worlds:
        print(f"Warning: Only {len(grid_files)} files found, using all available.")
        num_worlds = len(grid_files)

    print(f"Found {len(grid_files)} grid files:")
    for f in grid_files[:10]:
        print(f"  {f}")
    if len(grid_files) > 10:
        print(f"  ... and {len(grid_files) - 10} more files")

    for i in range(num_worlds):
        file_path = grid_files[i]
        relative_path = os.path.relpath(file_path, shelf_dir)
        shelf_name = os.path.splitext(relative_path)[0].replace(os.sep, '_')
        shelf_files.append(shelf_name)

        try:
            suffix = Path(file_path).suffix
            print(f"Loading {file_path}")

            # Load text-based grid
            if suffix in [".txt", ".csv"]:
                with open(file_path, 'r') as f:
                    lines = f.readlines()
                grid_data = []
                for line in lines:
                    if line.strip():
                        if ',' in line:
                            row = [int(float(x.strip())) for x in line.strip().split(',') if x.strip()]
                        else:
                            row = [int(float(x)) for x in line.strip().split() if x.strip()]
                        if row:
                            grid_data.append(row)
                if not grid_data:
                    raise ValueError("No valid data in text file")
                grid = np.array(grid_data, dtype=np.uint8)
                print(f"  Loaded text grid: {grid.shape}")

            # Load binary grid
            else:
                with open(file_path, 'rb') as f:
                    data = f.read()
                print(f"  Binary file size: {len(data)} bytes")

                if len(data) == 10000:
                    grid = np.frombuffer(data, dtype=np.uint8).reshape((100, 100))
                elif len(data) == 2500:
                    grid = np.frombuffer(data, dtype=np.uint8).reshape((50, 50))
                elif len(data) == 40000:
                    grid = np.frombuffer(data, dtype=np.uint32).reshape((100, 100))
                    grid = (grid > 0).astype(np.uint8) * 255
                elif len(data) == 160000:
                    grid = np.frombuffer(data, dtype=np.uint8).reshape((200, 200))
                else:
                    sqrt_size = int(math.sqrt(len(data)))
                    if sqrt_size * sqrt_size == len(data):
                        grid = np.frombuffer(data, dtype=np.uint8).reshape((sqrt_size, sqrt_size))
                        print(f"  Detected {sqrt_size}x{sqrt_size} grid")
                    else:
                        raise ValueError("Unknown binary format")

            # Normalize grid
            if grid.max() <= 1:
                grid *= 255
            elif grid.max() > 255:
                grid = ((grid - grid.min()) / (grid.max() - grid.min()) * 255).astype(np.uint8)

            free_ratio = np.sum(grid > 0) / grid.size
            print(f"  Grid shape: {grid.shape}, Free space: {free_ratio:.2%}")

            # Fix extreme free/obstacle ratios
            if free_ratio < 0.1 or free_ratio > 0.95:
                print(f"  Warning: Extreme free space ratio ({free_ratio:.2%}), regenerating grid")
                grid = np.ones(grid.shape, dtype=np.uint8) * 255
                for _ in range(int(grid.size * 0.25)):
                    x = random.randint(0, grid.shape[1] - 1)
                    y = random.randint(0, grid.shape[0] - 1)
                    grid[y, x] = 0

            worlds.append(grid)

        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            grid = np.ones((100, 100), dtype=np.uint8) * 255
            grid[20:80, 20:80] = 0
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
    """Main experimental pipeline for Random Edge Sampling RRT"""
    print("=" * 60)
    print("RANDOM EDGE SAMPLING RRT EXPERIMENTAL PIPELINE")
    print("=" * 60)
    
    # Create results directory
    results_dir = Path("edge_sampling_rrt_results")
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
        
        # Create walkable cells set for this world
        walkable_cells = create_walkable_cells(grid)
        print(f"  Found {len(walkable_cells)} walkable cells")
        
        # Generate 5 start-goal pairs
        pairs = generate_start_goal_pairs(grid, num_pairs=5)
        print(f"  Generated {len(pairs)} start-goal pairs")
        
        # Test each pair
        for pair_id, (start, goal) in enumerate(pairs):
            print(f"  Pair {pair_id + 1}: Start{start} -> Goal{goal}")
            
            # Run Random Edge Sampling RRT
            start_time = time.time()
            nodes, node_count = rrt_edge_sampling(grid, start, goal, walkable_cells, step_size=2, max_iter=5000)
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
                'algorithm': 'edge_sampling_rrt'
            }
            results.append(result)
    
    # Save results
    results_file = results_dir / "edge_sampling_rrt_results.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Print summary statistics
    print("\n" + "=" * 60)
    print("RANDOM EDGE SAMPLING RRT - FINAL RESULTS SUMMARY")
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