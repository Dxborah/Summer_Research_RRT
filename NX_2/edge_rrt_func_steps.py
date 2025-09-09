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

def update_visibility(seen_cells, visible_grid):
    """Updates visibility from a given position by casting rays"""
    if max_distance is None:
        max_distance = grid.shape[0]
        
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

def rrt_edge_exploration(grid, start, goal, walkable_cells, visibility_data, max_iter=5000, step_limit=3):
    """RRT implementation with edge of coverage exploration using Dijkstra,
    but moves step-by-step toward the target with a step limit per iteration.
    """
    global visible_grid, visited

    print("goal: ", goal)
    
    # Initialize RRT tree
    rrt = RRTNetworkX(grid, walkable_cells)
    #rrt.path_explored = []   # track orange path
    #rrt.path_explored.append(start)  # start position
    
    # Add start node
    start_x, start_y = start[0] + 0.5, start[1] + 0.5
    goal_x, goal_y = goal[0] + 0.5, goal[1] + 0.5
    
    # Current position (discrete grid coordinates)
    current_pos = start
    
    start_id = rrt.add_node(start_x, start_y)
    visited = set([start])
    
    current_node_id = start_id
    
    for iteration in range(max_iter):
        # Compute visibility from current visited set
        grid = np.ones((50, 50), dtype=int)
        og_vision, og_state, og_edge_seen = process_vision_clean(visited, grid.copy(), visibility_data)
            
        # Build graph based on current visibility
        visibility_graph = build_visibility_based_graph(grid, og_vision)

        # Decide target: goal if reachable soon, otherwise frontier
        goal_rechable = reachable_within_n_steps(visibility_graph, visited, goal, 3)
        if goal_rechable or len(og_edge_seen) == 0:
            target_cell = goal
        else:
            target_cell = random.choice(list(og_edge_seen))
        target_x, target_y = target_cell
        
        print(f"Iteration {iteration}: Current {current_pos}, Target {target_cell}")
        
        # Check if current position and target are in the graph
        if current_pos not in visibility_graph.nodes():
            print(f"Current position {current_pos} not in visibility graph")
            continue
            
        if target_cell not in visibility_graph.nodes():
            print(f"Target {target_cell} not in visibility graph")
            continue

        # Use Dijkstra to check if path exists
        distances = nx.single_source_dijkstra_path_length(
            visibility_graph, current_pos, weight='weight'
        )
        
        if target_cell in distances:
            closest_point = starting_point(visibility_graph, target_cell, visited)[0]
            # Path exists! Get the actual path
            full_path = nx.shortest_path(visibility_graph, closest_point, target_cell, weight='weight')
            print(f"Full path found with length {distances[target_cell]}: {full_path}")

            # Take only a limited number of steps along the path
            if len(full_path) > step_limit:
                step_path = full_path[:step_limit + 1]  # include node at the step limit
            else:
                step_path = full_path

            print(f"Taking step path: {step_path}")
            
            rrt.path_explored.append(step_path)

            # Add each step in the partial path to the RRT
            for step in step_path[1:]:  # skip the first (it's current_pos)
                step_x, step_y = step[0] + 0.5, step[1] + 0.5
                new_node_id = rrt.add_node(step_x, step_y, current_node_id)
                current_node_id = new_node_id

            # Update visibility and current position
            visited = visited | set(step_path)
            current_pos = step_path[-1]

            # Check if goal is reached
            if goal in visited:
                goal_id = rrt.add_node(goal_x, goal_y, current_node_id)
                print(f"Goal reached in {iteration + 1} iterations!")
                og_vision, og_state, og_edge_seen = process_vision_clean(visited, grid.copy(), visibility_data)
                return rrt, goal_id, og_state
                
        else:
            print(f"No path found to target {target_cell}")
            continue

    print("Maximum iterations reached without finding goal")
    og_vision, og_state, og_edge_seen = process_vision_clean(visited, grid.copy(), visibility_data)
    return rrt, None, og_state


def draw_result_on_image(grid, rrt, goal_id, visible_grid=None, filename="rrt_edge_exploration.png"):
    """Draw the RRT result with visibility information"""
    # Convert grayscale grid to BGR image
    img = np.stack([grid] * 3, axis=-1)
    img[grid == 255] = [255, 255, 255]  # White for free space
    img[grid == 0] = [0, 0, 0]          # Black for obstacles

    scale = 15
    img = cv2.resize(img, (grid.shape[1] * scale, grid.shape[0] * scale), 
                     interpolation=cv2.INTER_NEAREST)

    # Draw visibility information if provided
    if visible_grid is not None:
        visible_cells = get_visible_cells(visible_grid)
        edge_cells = compute_edge_of_coverage(grid, visible_grid, walkable_cells)
        
        # Draw visible cells in green
        for (x, y) in visible_cells:
            cx, cy = int((x + 0.5) * scale), int((y + 0.5) * scale)
            cv2.circle(img, (cx, cy), 3, (0, 255, 0), -1)  # Green for visible
            
        # Draw edge cells in blue
        for (x, y) in edge_cells:
            cx, cy = int((x + 0.5) * scale), int((y + 0.5) * scale)
            cv2.circle(img, (cx, cy), 3, (255, 0, 0), -1)  # Blue for frontier

    for path in rrt.path_explored:
        for traj_len in range(1, len(path)):
            x1, y1 = path[traj_len]
            x2, y2 = path[traj_len - 1]

            # Apply scaling like in visible_cells
            cx1, cy1 = int((x1 + 0.5) * scale), int((y1 + 0.5) * scale)
            cx2, cy2 = int((x2 + 0.5) * scale), int((y2 + 0.5) * scale)

            # Draw orange line
            cv2.line(img, (cx1, cy1), (cx2, cy2), (0, 165, 255), 2)

    cv2.imwrite(filename, img)
    print(f"Image saved: {filename}")

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

    rrt_tree, goal_node_id, og_state = rrt_edge_exploration(
    grid, start, goal, walkable_cells, visibility_map, max_iter=5000, step_limit=3)

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
        print(f"Path length: {total_steps} points")
        print(f"Execution time: {execution_time:.2f} seconds")
        
        if rrt_tree.tree.number_of_nodes() > 1:
            print(f"Tree depth: {nx.dag_longest_path_length(rrt_tree.tree)}")
        print(f"Tree edges: {rrt_tree.tree.number_of_edges()}")
        
        draw_result_on_image(grid, rrt_tree, goal_node_id, visible_grid, "rrt_edge_explo_step_success.png")
    else:
        print(f"FAILED - No path found in {max_iter} iterations")
        print(f"Execution time: {execution_time:.2f} seconds")
        draw_result_on_image(grid, rrt_tree, None, visible_grid, "rrt_edge_explo_step_failed.png")

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
