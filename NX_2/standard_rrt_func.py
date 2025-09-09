import cv2
import numpy as np
import networkx as nx
import random
import math
import json
from pathlib import Path
import shelve
import ast
import time

# -----------------
# RRT with NetworkX
# -----------------
class RRTNetworkX:
    def __init__(self, grid, walkable_cells):
        self.grid = grid
        self.walkable_cells = walkable_cells
        self.tree = nx.DiGraph()
        self.node_positions = {}
        self.path_explored = []
        self.node_counter = 0
        
    def add_node(self, x, y, parent_id=None):
        node_id = self.node_counter
        self.node_counter += 1
        self.tree.add_node(node_id, x=x, y=y)
        self.node_positions[node_id] = (x, y)
        if parent_id is not None:
            self.tree.add_edge(parent_id, node_id)
        return node_id
    
    def get_path_to_root(self, node_id):
        try:
            root_nodes = [n for n in self.tree.nodes() if self.tree.in_degree(n) == 0]
            if not root_nodes:
                return []
            root = root_nodes[0]
            path = nx.shortest_path(self.tree.reverse(), node_id, root)
            return [(self.tree.nodes[n]['x'], self.tree.nodes[n]['y']) for n in path]
        except nx.NetworkXNoPath:
            return []

def build_free_graph(grid, walkable_cells):
    """Graph connecting all walkable cells (8-connected)"""
    G = nx.Graph()
    for x, y in walkable_cells:
        G.add_node((x, y))
    height, width = grid.shape
    for x, y in walkable_cells:
        for dx, dy in [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]:
            nx_, ny_ = x+dx, y+dy
            if (0 <= nx_ < width) and (0 <= ny_ < height) and (nx_, ny_) in walkable_cells:
                G.add_edge((x,y), (nx_,ny_), weight=1)
    return G

def random_point(grid_size, goal=None):
    if goal and random.random() < 0.1:  # 10% goal bias
        return goal
    return (random.randint(0, grid_size-1), random.randint(0, grid_size-1))

def rrt_random_sample(grid, start, goal, walkable_cells, max_iter=1000):
    """RRT using NetworkX and Dijkstra but without edge-of-coverage"""
    rrt = RRTNetworkX(grid, walkable_cells)
    start_x, start_y = start[0]+0.5, start[1]+0.5
    goal_x, goal_y = goal[0]+0.5, goal[1]+0.5
    start_id = rrt.add_node(start_x, start_y)
    current_node_id = start_id
    visited = set([start])

    free_graph = build_free_graph(grid, walkable_cells)

    for i in range(max_iter):
        target = random_point(grid.shape[0], goal)
        if target not in free_graph:
            continue

        # Find closest visited node in free_graph
        reachable, _ = nx.single_source_dijkstra(free_graph, source=target)
        visited_targets = [v for v in visited if v in reachable]
        if not visited_targets:
            continue

        closest = min(visited_targets, key=lambda v: reachable[v])
        path = nx.shortest_path(free_graph, closest, target)
        rrt.path_explored.append(path)

        new_node_coords = (target[0]+0.5, target[1]+0.5)
        new_node_id = rrt.add_node(*new_node_coords, parent_id=current_node_id)
        current_node_id = new_node_id
        visited.update(path)

        if target == goal:
            goal_id = rrt.add_node(goal_x, goal_y, parent_id=current_node_id)
            return rrt, goal_id

    return rrt, None

def draw_result_on_image(grid, rrt, goal_id=None, filename="rrt_result.png"):
    img = np.stack([grid]*3, axis=-1)
    img[grid==255] = [255,255,255]
    img[grid==0] = [0,0,0]
    scale = 15
    img = cv2.resize(img, (grid.shape[1]*scale, grid.shape[0]*scale), interpolation=cv2.INTER_NEAREST)

    for path in rrt.path_explored:
        for i in range(1, len(path)):
            x1, y1 = path[i-1]
            x2, y2 = path[i]
            cv2.line(img, (int((x1+0.5)*scale), int((y1+0.5)*scale)),
                     (int((x2+0.5)*scale), int((y2+0.5)*scale)), (0,165,255), 2)

    cv2.imwrite(filename, img)
    print(f"Image saved: {filename}")

def run(seed, image, json_path, shelf_path, start, goal, max_iter=500, resize=(50,50)):
    random.seed(seed)
    np.random.seed(seed)

    map_img = cv2.imread(str(image), cv2.IMREAD_GRAYSCALE)
    if resize:
        map_img = cv2.resize(map_img, tuple(resize), interpolation=cv2.INTER_NEAREST)
    grid = np.where(map_img>127,255,0).astype(np.uint8)

    # Load visibility / walkable data
    if json_path and Path(json_path).exists():
        with open(json_path,'r') as f:
            visibility_map = json.load(f)
    else:
        visibility_data = shelve.open(shelf_path)
        for k in visibility_data.keys():
            visibility_data[k] = [tuple(x) if isinstance(x,(tuple,list)) else ast.literal_eval(x) for x in visibility_data[k]]
        json_visibility_data = {k:v for k,v in visibility_data.items()}
        with open(json_path,'w') as f:
            json.dump(json_visibility_data,f,indent=2)
        with open(json_path,'r') as f:
            visibility_map = json.load(f)

    walkable_cells = set(map(tuple, visibility_map["all"])) - set(map(tuple, visibility_map["blocked"]))

    start_time = time.time()
    rrt_tree, goal_id = rrt_random_sample(grid, start, goal, walkable_cells, max_iter=max_iter)
    end_time = time.time()
    execution_time = end_time - start_time

    final_path = None
    if goal_id is not None:
        final_path = rrt_tree.get_path_to_root(goal_id)
        print(f"SUCCESS: Goal reached!")
        draw_result_on_image(grid, rrt_tree, goal_id)
    else:
        print(f"FAILED: Goal not reached")
        draw_result_on_image(grid, rrt_tree)

    print(f"Execution time: {execution_time:.2f}s")
    return rrt_tree, goal_id, final_path, execution_time


if __name__ == "__main__":
    # Example start and goal
    start = (26, 21)
    goal = (40, 41)

    # Paths to your image and shelf/JSON files
    image_path = "/Users/deborahzeleke/Library/CloudStorage/OneDrive-McGillUniversity/summer_research/Summer_Research_RRT/NX_2/15000-15999/dungeon_2143.png"
    json_path = "/Users/deborahzeleke/Library/CloudStorage/OneDrive-McGillUniversity/summer_research/Summer_Research_RRT/NX_2/15000-15999/dungeon_2143.json"
    shelf_path = "/Users/deborahzeleke/Library/CloudStorage/OneDrive-McGillUniversity/summer_research/Summer_Research_RRT/NX_2/15000-15999/dungeon_2143_1.shelf"

    rrt_tree, goal_node_id, final_path, exec_time = run(
        seed=100,
        image=image_path,
        json_path=json_path,
        shelf_path=shelf_path,
        start=start,
        goal=goal,
        max_iter=500,
        resize=(50, 50)
    )

    if final_path:
        print(f"Final path length (nodes): {len(final_path)}")
    else:
        print("No path found to goal.")
