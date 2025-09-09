import cv2
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import math
import random
import json
import time
from collections import defaultdict
from pathlib import Path
import shelve
import ast
from sup_functions import *
from visuals import grid_plot_2, grid_plot_rrt

# Global tracking variables (re-initialized inside run())
visible_grid = None
visited = set()
walkable_cells = set()
grid = None

class RRTNetworkX:
    def __init__(self, grid, walkable_cells):
        self.grid = grid
        self.walkable_cells = walkable_cells
        self.tree = nx.DiGraph()  # Directed graph for parent-child relationships
        self.path_explored = [] #list of list of points taken
        self.node_positions = {}  # Maps node_id to (x, y) coordinates
        self.node_counter = 0
        
    def add_node(self, x, y, parent_id=None):
        """Add a node to the RRT tree"""
        node_id = self.node_counter
        self.node_counter += 1
        
        # Add node with position attributes
        self.tree.add_node(node_id, x=x, y=y)
        self.node_positions[node_id] = (x, y)
        
        # Add edge from parent if specified
        if parent_id is not None:
            self.tree.add_edge(parent_id, node_id)
            
        return node_id
    
    def get_path_to_root(self, node_id):
        """Get path from node to root using NetworkX"""
        try:
            # Find root node (node with no predecessors)
            root_nodes = [n for n in self.tree.nodes() if self.tree.in_degree(n) == 0]
            if not root_nodes:
                return []
            
            root = root_nodes[0]
            path = nx.shortest_path(self.tree.reverse(), node_id, root)
            
            # Convert node IDs to coordinates
            coord_path = [(self.tree.nodes[nid]['x'], self.tree.nodes[nid]['y']) for nid in path]
            return coord_path
            
        except nx.NetworkXNoPath:
            return []
    
    def get_all_edges(self):
        """Get all edges in the tree for visualization"""
        edges = []
        for parent_id, child_id in self.tree.edges():
            parent_pos = self.node_positions[parent_id]
            child_pos = self.node_positions[child_id]
            edges.append((parent_pos, child_pos))
        return edges

def starting_point(G, source, targets, weight="weight"):
    # Get all single-source distances
    dist = nx.single_source_dijkstra_path_length(G, source, weight=weight)
    # Filter only valid targets
    valid = {t: dist[t] for t in targets if t in dist}
    if not valid:
        return None, float("inf")  # no reachable target
    # Pick the one with minimum distance
    best_target = min(valid, key=valid.get)
    return best_target, valid[best_target]

def reachable_within_n_steps(G, sources, target, n, weight=None):
    """
    Return True iff there exists a path from ANY node in `sources` to `target`
    with cost (or steps if unweighted) <= n.
    """
    if target not in G:
        return False

    for s in sources:
        if s not in G:
            continue
        # Dijkstra from this source
        dist = nx.single_source_dijkstra_path_length(G, s, weight=weight, cutoff=n)
        if target in dist and dist[target] <= n:
            return True

    return False

def update_visibility(pos, visible_grid, max_distance=None):
    """Updates visibility from a given position by casting rays"""
    x, y = pos
    height, width = visible_grid.shape
    if max_distance is None:
        max_distance = max(height, width)

    num_rays = 360
    for angle in np.linspace(0, 2 * np.pi, num_rays, endpoint=False):
        for dist in range(1, max_distance):
            dx = x + dist * math.cos(angle)
            dy = y + dist * math.sin(angle)
            ix = int(math.floor(dx))
            iy = int(math.floor(dy))
            if 0 <= ix < width and 0 <= iy < height:
                visible_grid[iy, ix] = 1
                if grid[iy, ix] == 0:  # obstacle blocks
                    break
            else:
                break

def get_visible_cells(visible_grid):
    """Get set of all visible cells (green cells in your color scheme)"""
    visible_cells = set()
    height, width = visible_grid.shape
    for y in range(height):
        for x in range(width):
            if visible_grid[y, x] == 1:
                visible_cells.add((x, y))
    return visible_cells

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
            nx_, ny_ = x + dx, y + dy
            if (nx_, ny_) in walkable_cells and visible_grid[ny_, nx_] == 0:
                # Neighbor is white, walkable, and unseen — candidate edge
                edge_candidates.add((nx_, ny_))

    # Step 3: Finalize the edge by intersecting with unseen white cells
    edge_of_coverage = edge_candidates & unseen_white
    return edge_of_coverage

def find_270_corners(grid):
    # Convert to binary if needed
    binary_grid = (grid > 127).astype(int)  # Convert 255 to 1, 0 to 0
    rows, cols = binary_grid.shape
    corner_positions = []
    for i in range(rows - 1):
        for j in range(cols - 1):
            block = binary_grid[i:i+2, j:j+2]
            if block.sum() == 3:
                black_idx = np.argwhere(block == 0)[0]
                di, dj = 1 - black_idx[0], 1 - black_idx[1]
                corner_i, corner_j = i + di, j + dj
                corner_positions.append((corner_j, corner_i))  # Return as (x, y)
    return corner_positions

def build_visibility_based_graph(grid, relevant_cells):#visible_grid, walkable_cells):
    """Build a NetworkX graph only for visible cells and their frontier"""
    G = nx.Graph()
    height, width = grid.shape
    
    # Get visible cells and frontier
    # visible_cells = get_visible_cells(visible_grid)
    # frontier_cells = compute_edge_of_coverage(grid, visible_grid, walkable_cells)
    
    # Only consider cells that are either visible or on the frontier
    #relevant_cells = visible_cells | frontier_cells
    
    # Add nodes for relevant cells
    for (x, y) in relevant_cells:
        if (x, y) in walkable_cells:  # Double-check it's walkable
            G.add_node((x, y))
    
    # Add edges between relevant cells (8-connected)
    for (x, y) in relevant_cells:
        if (x, y) not in walkable_cells:
            continue
            
        for dx, dy in [(-1,0), (1,0), (0,-1), (0,1), (-1,-1), (-1,1), (1,-1), (1,1)]:
            nx_, ny_ = x + dx, y + dy
            
            # Check bounds and if neighbor is walkable and relevant
            if (0 <= nx_ < width and 0 <= ny_ < height and 
                (nx_, ny_) in walkable_cells and (nx_, ny_) in relevant_cells):
                G.add_edge((x, y), (nx_, ny_), weight=1)
    
    return G

'''
def rrt_corner_exploration(grid, start, goal, walkable_cells, visibility_data, max_iter=100):
    """
    RRT using 270° corners and edge-of-coverage strategy with max iterations:
    - Precompute all 270° corners
    - Move from start to the corner in visibility that maximizes new coverage
    - Repeat until goal is visible, then move directly to goal
    """
    global visible_grid, visited

    print("Starting 270° corner-based RRT exploration...")

    # Debug: Check grid values
    print(f"Grid shape: {grid.shape}")
    print(f"Grid unique values: {np.unique(grid)}")
    print(f"Grid dtype: {grid.dtype}")

    # Initialize RRT tree
    rrt = RRTNetworkX(grid, walkable_cells)

    # Add start node
    start_x, start_y = start[0] + 0.5, start[1] + 0.5
    current_node_id = rrt.add_node(start_x, start_y)
    current_pos = start
    visited = set([start])

    # Initialize visibility
    update_visibility(current_pos, visible_grid)

    # Precompute 270° corners
    corners = find_270_corners(grid)
    print(f"Found {len(corners)} total corners: {corners[:5] if len(corners) > 0 else 'None'}...")  # Debug print
    corners = set(corners)  # for fast lookup

    # Debug: Check initial visibility
    visible_cells = get_visible_cells(visible_grid)
    print(f"Visible cells from start: {len(visible_cells)}")
    print(f"First few visible cells: {list(visible_cells)[:5] if len(visible_cells) > 0 else 'None'}")
    
    # Debug: Check if any corners are in walkable cells
    walkable_corners = [c for c in corners if c in walkable_cells]
    print(f"Walkable corners: {len(walkable_corners)} out of {len(corners)}")
    print(f"First few walkable corners: {walkable_corners[:5] if len(walkable_corners) > 0 else 'None'}")

    # Build graph over all walkable cells (8-connected)
    G = nx.Graph()
    for (x, y) in walkable_cells:
        G.add_node((x, y))
    for (x, y) in walkable_cells:
        for dx, dy in [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]:
            nx_, ny_ = x+dx, y+dy
            if (nx_, ny_) in walkable_cells:
                G.add_edge((x, y), (nx_, ny_), weight=1)

    iterations = 0
    while iterations < max_iter:
        iterations += 1
        visible_cells = get_visible_cells(visible_grid)

        # If goal is visible, move directly to it
        if goal in visible_cells:
            path_to_goal = nx.shortest_path(G, source=current_pos, target=goal, weight='weight')
            prev_node_id = current_node_id
            for pos in path_to_goal[1:]:
                node_id = rrt.add_node(pos[0]+0.5, pos[1]+0.5, prev_node_id)
                prev_node_id = node_id
            current_node_id = prev_node_id
            visited.update(path_to_goal)
            print(f"Goal reached in {iterations} iterations!")
            return rrt, current_node_id, visited

        # Find visible corners
        visible_corners = [c for c in corners if c in visible_cells]
        print(f"Iteration {iterations}: Found {len(visible_corners)} visible corners out of {len(corners)} total corners")  # Debug print
        
        if not visible_corners:
            print("No visible 270° corners left, cannot continue")
            print(f"Debug: Total visible cells: {len(visible_cells)}")  # Debug print
            print(f"Debug: Sample visible cells: {list(visible_cells)[:10] if len(visible_cells) > 0 else 'None'}")  # Debug print
            print(f"Debug: All corners: {list(corners)[:10] if len(corners) > 0 else 'None'}")  # Debug print
            return rrt, None, visited

        # Pick corner that sees the most NEW edge-of-coverage cells
        def evaluate_corner_coverage(corner):
            # Simulate visibility from this corner and count new cells
            temp_visible_grid = visible_grid.copy()
            update_visibility(corner, temp_visible_grid)
            new_edge = compute_edge_of_coverage(grid, temp_visible_grid, walkable_cells)
            current_edge = compute_edge_of_coverage(grid, visible_grid, walkable_cells)
            return len(new_edge - current_edge)
        
        best_corner = max(visible_corners, key=evaluate_corner_coverage)
        
        # Debug: Show corner selection info
        current_edge_count = len(compute_edge_of_coverage(grid, visible_grid, walkable_cells))
        best_new_coverage = evaluate_corner_coverage(best_corner)
        print(f"  Selected corner {best_corner} with {best_new_coverage} new edge cells (current edge: {current_edge_count})")

        # Move to that corner via shortest path in graph
        path_to_corner = nx.shortest_path(G, source=current_pos, target=best_corner, weight='weight')

        # Add path to RRT tree
        prev_node_id = current_node_id
        for pos in path_to_corner[1:]:
            node_id = rrt.add_node(pos[0]+0.5, pos[1]+0.5, prev_node_id)
            prev_node_id = node_id
        current_node_id = prev_node_id

        # Update visited and visibility
        visited.update(path_to_corner)
        current_pos = best_corner
        update_visibility(current_pos, visible_grid)

    print(f"Maximum iterations ({max_iter}) reached without seeing goal.")
    return rrt, None, visited
'''

def rrt_corner_exploration_with_visited_start(grid, start, goal, walkable_cells, visibility_data, max_iter=100):
    """
    Modified RRT using 270° corners but starting paths from closest visited cells:
    - Precompute all 270° corners
    - Find the corner that maximizes new coverage
    - Start path from the visited cell closest to that corner (instead of current position)
    - This should reduce zigzagging while maintaining corner-based exploration
    """
    global visible_grid, visited

    print("Starting modified 270° corner-based RRT exploration...")

    # Initialize RRT tree
    rrt = RRTNetworkX(grid, walkable_cells)

    # Add start node
    start_x, start_y = start[0] + 0.5, start[1] + 0.5
    current_node_id = rrt.add_node(start_x, start_y)
    visited = set([start])

    # Initialize visibility
    update_visibility(start, visible_grid)

    # Precompute 270° corners
    corners = find_270_corners(grid)
    print(f"Found {len(corners)} total corners")
    corners = set(corners)

    # Build graph over all walkable cells (8-connected)
    G = nx.Graph()
    for (x, y) in walkable_cells:
        G.add_node((x, y))
    for (x, y) in walkable_cells:
        for dx, dy in [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]:
            nx_, ny_ = x+dx, y+dy
            if (nx_, ny_) in walkable_cells:
                G.add_edge((x, y), (nx_, ny_), weight=1)

    iterations = 0
    while iterations < max_iter:
        iterations += 1
        visible_cells = get_visible_cells(visible_grid)

        # If goal is visible, move directly to it from closest visited cell
        if goal in visible_cells:
            closest_to_goal, _ = starting_point(G, goal, visited)
            if closest_to_goal is not None:
                path_to_goal = nx.shortest_path(G, source=closest_to_goal, target=goal, weight='weight')
                
                # Add path to RRT tree
                # Find the node corresponding to closest_to_goal
                closest_node_id = None
                for node_id in rrt.tree.nodes():
                    node_x = int(rrt.tree.nodes[node_id]['x'] - 0.5)
                    node_y = int(rrt.tree.nodes[node_id]['y'] - 0.5)
                    if (node_x, node_y) == closest_to_goal:
                        closest_node_id = node_id
                        break
                
                # If we can't find the node, use current_node_id
                if closest_node_id is None:
                    closest_node_id = current_node_id
                
                prev_node_id = closest_node_id
                for pos in path_to_goal[1:]:
                    node_id = rrt.add_node(pos[0]+0.5, pos[1]+0.5, prev_node_id)
                    prev_node_id = node_id
                
                rrt.path_explored.append(path_to_goal)
                visited.update(path_to_goal)
                print(f"Goal reached in {iterations} iterations!")
                return rrt, prev_node_id, visited
            
        # Find visible corners
        visible_corners = [c for c in corners if c in visible_cells]
        print(f"Iteration {iterations}: Found {len(visible_corners)} visible corners")
        
        if not visible_corners:
            print("No visible 270° corners left, cannot continue")
            return rrt, None, visited

        # Pick corner that sees the most NEW edge-of-coverage cells
        def evaluate_corner_coverage(corner):
            temp_visible_grid = visible_grid.copy()
            update_visibility(corner, temp_visible_grid)
            new_edge = compute_edge_of_coverage(grid, temp_visible_grid, walkable_cells)
            current_edge = compute_edge_of_coverage(grid, visible_grid, walkable_cells)
            return len(new_edge - current_edge)
        
        best_corner = max(visible_corners, key=evaluate_corner_coverage)
        
        # KEY CHANGE: Find closest visited cell to the best corner
        closest_visited_cell, distance = starting_point(G, best_corner, visited)
        
        if closest_visited_cell is None:
            print(f"No path found from visited cells to corner {best_corner}")
            continue
            
        print(f"  Selected corner {best_corner}")
        print(f"  Starting from visited cell {closest_visited_cell} (distance: {distance})")

        # Move from closest visited cell to that corner
        path_to_corner = nx.shortest_path(G, source=closest_visited_cell, target=best_corner, weight='weight')

        # Find the RRT node corresponding to closest_visited_cell
        start_node_id = None
        for node_id in rrt.tree.nodes():
            node_x = int(rrt.tree.nodes[node_id]['x'] - 0.5)
            node_y = int(rrt.tree.nodes[node_id]['y'] - 0.5)
            if (node_x, node_y) == closest_visited_cell:
                start_node_id = node_id
                break
        
        # If we can't find the exact node, use current_node_id as fallback
        if start_node_id is None:
            start_node_id = current_node_id

        # Add path to RRT tree
        prev_node_id = start_node_id
        for pos in path_to_corner[1:]:  # Skip first position (already in tree)
            node_id = rrt.add_node(pos[0]+0.5, pos[1]+0.5, prev_node_id)
            prev_node_id = node_id
        current_node_id = prev_node_id

        # Update visited and visibility
        rrt.path_explored.append(path_to_corner)
        visited.update(path_to_corner)
        update_visibility(best_corner, visible_grid)

    print(f"Maximum iterations ({max_iter}) reached without seeing goal.")
    return rrt, None, visited

def draw_result_on_image(grid, rrt, goal_id, visible_grid=None, filename="rrt_edge_exploration.png", 
                        start=None, goal=None):
    """Enhanced RRT visualization with better colors and final path"""
    # Convert grayscale grid to BGR image
    img = np.stack([grid] * 3, axis=-1)
    img[grid == 255] = [240, 240, 240]  # Light gray for free space
    img[grid == 0] = [1, 1, 1]       # Dark gray for obstacles

    scale = 15
    img = cv2.resize(img, (grid.shape[1] * scale, grid.shape[0] * scale), 
                     interpolation=cv2.INTER_NEAREST)

    # Draw visibility information if provided
    if visible_grid is not None:
        visible_cells = get_visible_cells(visible_grid)
        edge_cells = compute_edge_of_coverage(grid, visible_grid, walkable_cells)
        
        # Draw visible cells in light green
        for (x, y) in visible_cells:
            cx, cy = int((x + 0.5) * scale), int((y + 0.5) * scale)
            cv2.circle(img, (cx, cy), 2, (144, 238, 144), -1)  # Light green
            
        # Draw edge cells in bright blue
        for (x, y) in edge_cells:
            cx, cy = int((x + 0.5) * scale), int((y + 0.5) * scale)
            cv2.circle(img, (cx, cy), 3, (255, 100, 0), -1)  # Bright blue

    # Draw RRT tree edges in light gray
    all_edges = rrt.get_all_edges()
    for (parent_pos, child_pos) in all_edges:
        x1, y1 = parent_pos
        x2, y2 = child_pos
        cx1, cy1 = int(x1 * scale), int(y1 * scale)
        cx2, cy2 = int(x2 * scale), int(y2 * scale)
        cv2.line(img, (cx1, cy1), (cx2, cy2), (180, 180, 180), 1)

    # Draw exploration paths in orange
    for path in rrt.path_explored:
        for traj_len in range(1, len(path)):
            x1, y1 = path[traj_len]
            x2, y2 = path[traj_len - 1]
            cx1, cy1 = int((x1 + 0.5) * scale), int((y1 + 0.5) * scale)
            cx2, cy2 = int((x2 + 0.5) * scale), int((y2 + 0.5) * scale)
            cv2.line(img, (cx1, cy1), (cx2, cy2), (0, 165, 255), 3)  # Thicker orange line

    # Draw final path in thick red if goal reached
    if goal_id is not None:
        final_path = rrt.get_path_to_root(goal_id)
        if len(final_path) > 1:
            for i in range(1, len(final_path)):
                x1, y1 = final_path[i-1]
                x2, y2 = final_path[i]
                cx1, cy1 = int((x1 + 0.5) * scale), int((y1 + 0.5) * scale)
                cx2, cy2 = int((x2 + 0.5) * scale), int((y2 + 0.5) * scale)
                cv2.line(img, (cx1, cy1), (cx2, cy2), (0, 0, 255), 4)  # Thick red

    # Draw start and goal markers
    if start is not None:
        cx, cy = int((start[0] + 0.5) * scale), int((start[1] + 0.5) * scale)
        cv2.circle(img, (cx, cy), 6, (0, 255, 0), -1)  # Green start
        cv2.circle(img, (cx, cy), 8, (0, 200, 0), 2)

    if goal is not None:
        cx, cy = int((goal[0] + 0.5) * scale), int((goal[1] + 0.5) * scale)
        cv2.circle(img, (cx, cy), 6, (0, 0, 255), -1)  # Red goal
        cv2.circle(img, (cx, cy), 8, (0, 0, 200), 2)

    cv2.imwrite(filename, img)
    print(f"Enhanced image saved: {filename}")


# ---------- NEW: single entry-point function ----------
def run(seed, image, json_path, shelf_path, start, goal, max_iter=100, resize=(50, 50)):
    global visible_grid, visited, walkable_cells, grid

    # Seed
    SEED = int(seed)
    random.seed(SEED)
    np.random.seed(SEED)
    print(f"Using seed: {SEED}")

    # Load grid from image
    map_img = cv2.imread(str(image), cv2.IMREAD_GRAYSCALE)
    if map_img is None:
        raise FileNotFoundError(f"Could not read image: {image}")
    if resize is not None:
        map_img = cv2.resize(map_img, tuple(resize), interpolation=cv2.INTER_NEAREST)
    grid = np.where(map_img > 127, 255, 0).astype(np.uint8)
    grid_size = grid.shape[0]

    # Load visibility map from JSON
    if json_path is not None and Path(str(json_path)).exists():
        with open(str(json_path), 'r') as f:
            visibility_map = json.load(f)
    else:
        # assume json_path is actually a shelf path without extension
        # with shelve.open(str(shelf_path)) as db:
        #     ks = list(db.keys())
        #     print("num keys:", len(ks))
        #     print("first 20 keys:", ks[:20])


        
        visibility_data = shelve.open(shelf_path)
        # save as JSON next to the shelf

        non_empty_key = [
            k for k in list(visibility_data.keys())
            if visibility_data[k] != [] and k != "all" and k != "blocked"
        ]
        
        empty_key = [
            k for k in list(visibility_data.keys())
            if visibility_data[k] == [] and k != "all" and k != "blocked"
        ]

        visibility_data['all'] = non_empty_key + empty_key
        visibility_data['blocked'] = empty_key
        
        for k in list(visibility_data.keys()):
            visibility_data[k] = [
                tuple(x) if isinstance(x, (tuple, list)) else ast.literal_eval(x)
                for x in visibility_data[k]
            ]

        json_visibility_data = {}
        for visibility_data_key in list(visibility_data.keys()):
            json_visibility_data[visibility_data_key] = visibility_data[visibility_data_key]
        
        with open(json_path, "w") as f:
            json.dump(json_visibility_data, f, indent=2)
        # reopen as JSON (to be consistent)
        with open(json_path, "r") as f:
            visibility_map = json.load(f)
    visibility_map = {k: {tuple(x) for x in v} for k, v in visibility_map.items()}


    # Convert JSON coordinates to set of walkable and blocked cells
    all_cells = set(map(tuple, visibility_map["all"]))
    blocked_cells = set(map(tuple, visibility_map["blocked"]))
    walkable_cells = all_cells - blocked_cells

    # Reset globals for a fresh run
    visible_grid = np.zeros_like(grid, dtype=np.uint8)
    visited = set()

    # Safety check
    if grid[start[1], start[0]] == 0 or grid[goal[1], goal[0]] == 0:
        raise ValueError("Start or goal is inside an obstacle.")

    print("Running RRT with edge exploration and Dijkstra path planning...")
    start_time = time.time()

    #rrt_tree, goal_node_id, og_state = rrt_corner_exploration(grid, start, goal, walkable_cells, visibility_map, max_iter=max_iter)
    rrt_tree, goal_node_id, og_state = rrt_corner_exploration_with_visited_start(grid, start, goal, walkable_cells, visibility_map, max_iter=max_iter)

    end_time = time.time()
    execution_time = end_time - start_time

    total_steps = 0
    for mid_path in rrt_tree.path_explored:
        total_steps += len(mid_path) - 1


    final_path = None
    if goal_node_id is not None:
        final_path = rrt_tree.get_path_to_root(goal_node_id)
        print(f"SUCCESS - Goal reached!")
        print(f"Number of RRT nodes: {rrt_tree.tree.number_of_nodes()}")
        print(f"Path length: {total_steps} steps")
        print(f"Execution time: {execution_time:.2f} seconds")
        
        if rrt_tree.tree.number_of_nodes() > 1:
            print(f"Tree depth: {nx.dag_longest_path_length(rrt_tree.tree)}")
        print(f"Tree edges: {rrt_tree.tree.number_of_edges()}")
        
        draw_result_on_image(grid, rrt_tree, goal_node_id, visible_grid, "rrt_corner_explo_success.png", start, goal)
    else:
        print(f"FAILED - No path found in {max_iter} iterations")
        print(f"Execution time: {execution_time:.2f} seconds")
        draw_result_on_image(grid, rrt_tree, None, visible_grid, "rrt_corner_explo_failed.png", start, goal)

    return rrt_tree, goal_node_id, final_path, execution_time

# ---------- Original main preserved, now uses run() ----------
if __name__ == "__main__":
    start = (26,21)
    goal = (40,41)
    rrt_tree, goal_node_id, final_path, execution_time = run(
        seed=100,
        image="/Users/deborahzeleke/Library/CloudStorage/OneDrive-McGillUniversity/summer_research/Summer_Research_RRT/NX_2/15000-15999/dungeon_2143.png",
        json_path="/Users/deborahzeleke/Library/CloudStorage/OneDrive-McGillUniversity/summer_research/Summer_Research_RRT/NX_2/15000-15999/dungeon_2143.json",
        shelf_path="/Users/deborahzeleke/Library/CloudStorage/OneDrive-McGillUniversity/summer_research/Summer_Research_RRT/NX_2/15000-15999/dungeon_2143_1.shelf",
        start=start,
        goal=goal,
        max_iter=100,
        resize=(50, 50),
    )