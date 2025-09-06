import shelve
import numpy as np
from PIL import Image
from pathlib import Path
import networkx as nx
import matplotlib.pyplot as plt

def find_270_corners(grid):
    rows, cols = grid.shape
    corner_positions = []
    for i in range(rows - 1):
        for j in range(cols - 1):
            block = grid[i:i+2, j:j+2]
            if block.sum() == 3:
                black_idx = np.argwhere(block == 0)[0]
                di, dj = 1 - black_idx[0], 1 - black_idx[1]
                corner_i, corner_j = i + di, j + dj
                corner_positions.append((corner_i, corner_j))
    return corner_positions

def find_shelf_files(shelves_dir, dungeon_prefix):
    """Find shelf files by looking for .shelf.dat files and extracting base names"""
    path_obj = Path(shelves_dir)
    
    # Look for .shelf.dat files (these indicate the presence of a shelf)
    dat_files = list(path_obj.rglob(f"{dungeon_prefix}_*.shelf.dat"))
    
    # Extract the base shelf names (without .dat extension)
    shelf_bases = []
    for dat_file in dat_files:
        # Remove the .dat extension to get the shelf base name
        shelf_base = str(dat_file)[:-4]  # Remove '.dat'
        shelf_bases.append(shelf_base)
    
    return sorted(shelf_bases)

def explore_world_all_shelves(world_png, shelves_dir, dungeon_prefix, output_png):
    world = np.array(Image.open(world_png).convert("L")) // 255

    # Find shelf files using the corrected method
    shelf_files = find_shelf_files(shelves_dir, dungeon_prefix)
    
    if not shelf_files:
        print(f"No shelves found for {dungeon_prefix}")
        return

    print(f"Found {len(shelf_files)} shelf files:")
    for shelf_file in shelf_files:
        print(f"  {shelf_file}")

    merged_visibility = {}
    for shelf_path in shelf_files:
        print(f"Processing shelf: {shelf_path}")
        try:
            with shelve.open(shelf_path) as shelf_data:
                key_count = 0
                for key, value in shelf_data.items():
                    key_count += 1
                    if key not in merged_visibility:
                        merged_visibility[key] = value
                    else:
                        merged_visibility[key] = list({tuple(cell) for cell in merged_visibility[key] + value})
                print(f"  Processed {key_count} keys from {shelf_path}")
        except Exception as e:
            print(f"  Error reading shelf {shelf_path}: {e}")

    corners = find_270_corners(world)
    print(f"Found {len(corners)} corners in world")
    print(f"Total visibility keys: {len(merged_visibility)}")
    
    seen = np.zeros_like(world)

    visited_corners = set()
    processed_corners = 0
    
    while seen.sum() < world.size:
        corner = None
        for c in corners:
            if c not in visited_corners:
                corner = c
                break
        if corner is None:
            print("No more unvisited corners")
            break

        visited_corners.add(corner)
        key = f"({corner[1]}, {corner[0]})"
        if key in merged_visibility:
            cells_added = 0
            for cell in merged_visibility[key]:
                if seen[cell[0], cell[1]] == 0:  # Only count new cells
                    cells_added += 1
                seen[cell[0], cell[1]] = 1
            processed_corners += 1
            if processed_corners % 10 == 0:  # Progress update every 10 corners
                print(f"Processed {processed_corners} corners, seen {seen.sum()}/{world.size} cells")

    print(f"Final: Processed {processed_corners} corners, seen {seen.sum()}/{world.size} cells")
    
    # Create RGB image for red corner dots
    # Convert seen to proper colors: seen areas (1) become black (0), unseen areas (0) become white (255)
    base_img = seen * 255
    
    # First, mark corners on the original image before any transformations
    corner_img = np.stack([base_img, base_img, base_img], axis=-1)
    
    # Add corner visualization on original coordinates (swap x and y to match shelf keys)
    for corner in corners:
        row, col = corner
        # Swap coordinates to match how we're looking them up in shelves
        swapped_row, swapped_col = col, row
        if 0 <= swapped_row < corner_img.shape[0] and 0 <= swapped_col < corner_img.shape[1]:
            corner_img[swapped_row, swapped_col] = [255, 0, 0]  # Red dot for corner
    
    # Apply transformations to the entire image (including the marked corners)
    corner_img_flipped = np.flipud(corner_img)  # Flip vertically first
    corner_img_final = np.rot90(corner_img_flipped, k=-1)  # Then rotate 90° clockwise
    
    Image.fromarray(corner_img_final.astype(np.uint8)).save(output_png)
    print(f"Explored world saved to {output_png}")

def find_shortest_path_networkx(world_png, shelves_dir, dungeon_prefix, output_png):
    """
    Find shortest path between two random 270-degree corners that can see each other
    """
    # Load the world image
    world = np.array(Image.open(world_png).convert("L")) // 255
    rows, cols = world.shape
    
    print(f"World size: {rows} x {cols}")
    
    # Load visibility data from shelves
    shelf_files = find_shelf_files(shelves_dir, dungeon_prefix)
    if not shelf_files:
        print(f"No shelves found for {dungeon_prefix}")
        return None

    merged_visibility = {}
    for shelf_path in shelf_files:
        try:
            with shelve.open(shelf_path) as shelf_data:
                for key, value in shelf_data.items():
                    if key not in merged_visibility:
                        merged_visibility[key] = value
                    else:
                        merged_visibility[key] = list({tuple(cell) for cell in merged_visibility[key] + value})
        except Exception as e:
            print(f"Error reading shelf {shelf_path}: {e}")

    # Find 270-degree corners
    corners = find_270_corners(world)
    print(f"Found {len(corners)} corners in world")
    
    # Find pairs of corners that can see each other
    visible_pairs = []
    for i, corner1 in enumerate(corners):
        key1 = f"({corner1[1]}, {corner1[0]})"  # Swapped coordinates
        if key1 in merged_visibility:
            visible_cells1 = set(tuple(cell) for cell in merged_visibility[key1])
            
            for j, corner2 in enumerate(corners[i+1:], i+1):
                key2 = f"({corner2[1]}, {corner2[0]})"  # Swapped coordinates  
                if key2 in merged_visibility:
                    # Check if corner2 can see corner1 (corner1 is in corner2's visibility)
                    if (corner1[0], corner1[1]) in visible_cells1 or (corner2[0], corner2[1]) in set(tuple(cell) for cell in merged_visibility[key2]):
                        visible_pairs.append((corner1, corner2))
    
    print(f"Found {len(visible_pairs)} pairs of corners that can see each other")
    
    if not visible_pairs:
        print("No corners can see each other - using random corners instead")
        if len(corners) < 2:
            print("Need at least 2 corners")
            return None
        # Fallback to random corners
        import random
        start_point, end_point = random.sample(corners, 2)
    else:
        # Select a random pair from visible pairs
        import random
        start_point, end_point = random.choice(visible_pairs)
    
    print(f"Start point: {start_point}")
    print(f"End point: {end_point}")
    
    # Create NetworkX grid graph
    print("Creating NetworkX grid graph...")
    G = nx.grid_2d_graph(rows, cols)
    
    # Remove nodes that correspond to walls
    nodes_to_remove = [(i, j) for i in range(rows) for j in range(cols) if world[i, j] == 0]
    G.remove_nodes_from(nodes_to_remove)
    print(f"Graph has {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
    
    # Find shortest path using Dijkstra's algorithm
    print("Finding shortest path...")
    try:
        path = nx.shortest_path(G, source=start_point, target=end_point, method='dijkstra')
        path_length = len(path) - 1
        print(f"Found path with {path_length} steps")
    except nx.NetworkXNoPath:
        print("No path found between the corners!")
        return None
    except Exception as e:
        print(f"Error finding path: {e}")
        return None
    
    # Create visualization
    print("Creating visualization...")
    
    # Create RGB image: walls = black, walkable = white
    result_img = np.stack([world * 255, world * 255, world * 255], axis=-1)
    
    # Mark the path in blue
    for row, col in path:
        result_img[row, col] = [0, 0, 255]  # Blue for path
    
    # Mark start point in green
    result_img[start_point[0], start_point[1]] = [0, 255, 0]  # Green for start
    
    # Mark end point in red  
    result_img[end_point[0], end_point[1]] = [255, 0, 0]  # Red for end
    
    # Save the result
    Image.fromarray(result_img.astype(np.uint8)).save(output_png)
    print(f"Path visualization saved to {output_png}")
    
    return path

def get_random_walkable_points(world_png, num_points=2):
    """Get random walkable points from the world image"""
    world = np.array(Image.open(world_png).convert("L")) // 255
    walkable_cells = np.argwhere(world == 1)
    
    if len(walkable_cells) < num_points:
        print(f"Error: Only {len(walkable_cells)} walkable cells found, need {num_points}")
        return None
    
    indices = np.random.choice(len(walkable_cells), size=num_points, replace=False)
    points = [tuple(walkable_cells[i]) for i in indices]
    return points

def get_corner_walkable_points(world_png):
    """Get walkable points near the corners of the walkable area"""
    world = np.array(Image.open(world_png).convert("L")) // 255
    walkable_cells = np.argwhere(world == 1)
    
    if len(walkable_cells) < 2:
        print("Error: Need at least 2 walkable cells")
        return None
    
    # Find bounding box of walkable area
    min_row, min_col = walkable_cells.min(axis=0)
    max_row, max_col = walkable_cells.max(axis=0)
    
    # Try corners
    corner_candidates = [
        (min_row, min_col),     # Top-left
        (min_row, max_col),     # Top-right  
        (max_row, min_col),     # Bottom-left
        (max_row, max_col),     # Bottom-right
    ]
    
    # Find closest walkable cells to corners
    valid_points = []
    for corner in corner_candidates:
        distances = np.sum((walkable_cells - corner) ** 2, axis=1)
        closest_idx = np.argmin(distances)
        valid_points.append(tuple(walkable_cells[closest_idx]))
    
    # Remove duplicates and return first two distinct points
    unique_points = list(dict.fromkeys(valid_points))  # Preserves order
    return unique_points[:2] if len(unique_points) >= 2 else None

# Example usage of the original function
explore_world_all_shelves(
    world_png="../divided_world_1/15000-15999/dungeon_2143.png",
    shelves_dir="../divided_world_1",
    dungeon_prefix="dungeon_2143",
    output_png="explored_dungeon_2143.png"
)

# Find shortest path between random 270-degree corners that can see each other
path = find_shortest_path_networkx(
    world_png="../divided_world_1/15000-15999/dungeon_2143.png",
    shelves_dir="../divided_world_1", 
    dungeon_prefix="dungeon_2143",
    output_png="pathfinding_corners2.png"
)