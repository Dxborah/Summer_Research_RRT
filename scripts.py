import cv2
import numpy as np
import matplotlib.pyplot as plt
import math
import random
import heapq
import json
import time
import os

# Global state used in RRT
visible_grid = None
pheromone_grid = None
visited = None


# Load shared resources once
map_img = cv2.imread("IMG_8864.png", cv2.IMREAD_GRAYSCALE)
map_img = cv2.resize(map_img, (50, 50), interpolation=cv2.INTER_NEAREST)
base_grid = np.where(map_img > 127, 255, 0).astype(np.uint8)
grid_size = base_grid.shape[0]

with open("FILE_3036.json", "r") as f:
    visibility_map_raw = json.load(f)

all_cells = set(map(tuple, visibility_map_raw["all"]))
blocked_cells = set(map(tuple, visibility_map_raw["blocked"]))
walkable_cells = all_cells - blocked_cells

start = (8, 21)
goal = (30, 16)

def run_rrt_v1(seed):
    import random
    random.seed(seed)
    np.random.seed(seed)

    grid = base_grid.copy()

    global visible_grid, pheromone_grid, visited
    visible_grid = np.zeros_like(grid, dtype=np.uint8)
    pheromone_grid = np.zeros_like(grid, dtype=np.float32)
    visited = set()

    class Node:
        def __init__(self, x, y):
            self.x = x + 0.5
            self.y = y + 0.5
            self.parent = None

    def grid_value(x, y):
        ix = int(math.floor(x))
        iy = int(math.floor(y))
        if 0 <= iy < grid.shape[0] and 0 <= ix < grid.shape[1]:
            return grid[iy, ix]
        return 0

    def distance_angle(x1, y1, x2, y2):
        return math.hypot(x2 - x1, y2 - y1), math.atan2(y2 - y1, x2 - x1)

    def nearest_node(node_list, x, y):
        return min(node_list, key=lambda node: math.hypot(node.x - x, node.y - y))

    def compute_edge_of_coverage(grid, visible_grid, walkable_cells):
        seen_white = set()
        edge_candidates = set()
        unseen_white = set()
        for (x, y) in walkable_cells:
            if visible_grid[y, x] == 1:
                seen_white.add((x, y))
            else:
                unseen_white.add((x, y))
        for (x, y) in seen_white:
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nx, ny = x + dx, y + dy
                if (nx, ny) in walkable_cells and visible_grid[ny, nx] == 0:
                    edge_candidates.add((nx, ny))
        return edge_candidates & unseen_white

    def ant_random_point(goal, goal_sample_rate=0.05):
        if random.random() < goal_sample_rate:
            return goal
        edge_of_coverage = compute_edge_of_coverage(grid, visible_grid, walkable_cells)
        if not edge_of_coverage:
            return random.randint(0, grid_size - 1), random.randint(0, grid_size - 1)
        edge_list = list(edge_of_coverage)
        weights = [pheromone_grid[y, x] + 1e-6 for x, y in edge_list]
        probs = np.array(weights, dtype=np.float32)
        probs /= probs.sum()
        sampled_index = np.random.choice(len(edge_list), p=probs)
        return edge_list[sampled_index]

    def steer(from_node, to_x, to_y, step_size):
        dist, angle = distance_angle(from_node.x, from_node.y, to_x + 0.5, to_y + 0.5)
        step = min(step_size, dist)
        for i in range(1, int(step) + 1):
            nx = from_node.x + i * math.cos(angle)
            ny = from_node.y + i * math.sin(angle)
            if grid_value(nx, ny) == 0:
                return None
        return from_node.x + step * math.cos(angle), from_node.y + step * math.sin(angle)

    def interpolate_path(x1, y1, x2, y2):
        x1, y1 = int(math.floor(x1)), int(math.floor(y1))
        x2, y2 = int(math.floor(x2)), int(math.floor(y2))
        if grid[y1, x1] == 0 or grid[y2, x2] == 0:
            return []
        start, goal = (x1, y1), (x2, y2)
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
                if 0 <= nx < grid.shape[1] and 0 <= ny < grid.shape[0] and grid[ny, nx] != 0:
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
        return path[::-1]

    def collision(x1, y1, x2, y2):
        return len(interpolate_path(x1, y1, x2, y2)) == 0

    def update_visibility(x, y, max_distance=grid_size):
        for angle in np.linspace(0, 2 * np.pi, 360, endpoint=False):
            for dist in range(1, max_distance):
                dx = x + dist * math.cos(angle)
                dy = y + dist * math.sin(angle)
                ix, iy = int(math.floor(dx)), int(math.floor(dy))
                if 0 <= ix < grid.shape[1] and 0 <= iy < grid.shape[0]:
                    visible_grid[iy, ix] = 1
                    if grid[iy, ix] == 0:
                        break
                else:
                    break

    def rrt(grid, start, goal, step_size=1, max_iter=5000):
        global visible_grid, pheromone_grid
        start_node = Node(*start)
        goal_node = Node(*goal)
        update_visibility(start_node.x, start_node.y)
        nodes = [start_node]
        visited = set([(int(start_node.x), int(start_node.y))])
        for _ in range(max_iter):
            rand_x, rand_y = ant_random_point(goal)
            nearest = nearest_node(nodes, rand_x + 0.5, rand_y + 0.5)
            steered = steer(nearest, rand_x, rand_y, step_size)
            if steered:
                new_x, new_y = steered
                ix, iy = int(math.floor(new_x)), int(math.floor(new_y))
                if (ix, iy) not in visited and not collision(nearest.x, nearest.y, new_x, new_y):
                    new_node = Node(ix, iy)
                    new_node.x, new_node.y = new_x, new_y
                    new_node.parent = nearest
                    nodes.append(new_node)
                    visited.add((ix, iy))
                    update_visibility(new_x, new_y)
                    pheromone_grid *= 0.998
                    if abs(new_x - goal_node.x) + abs(new_y - goal_node.y) <= 3:
                        goal_node.parent = new_node
                        nodes.append(goal_node)
                        return nodes, len(nodes), True
        return [], 0, False

    if grid[start[1], start[0]] == 0 or grid[goal[1], goal[0]] == 0:
        return 0, 0.0, False

    start_time = time.time()
    nodes, steps, success = rrt(grid, start, goal)
    end_time = time.time()
    return steps, end_time - start_time, success




def run_rrt_v2(seed):
    random.seed(seed)
    np.random.seed(seed)

    grid = base_grid.copy()
    visible_grid = np.zeros_like(grid, dtype=np.uint8)
    pheromone_grid = np.zeros_like(grid, dtype=np.float32)

    class Node:
        def __init__(self, x, y):
            self.x = x + 0.5
            self.y = y + 0.5
            self.parent = None

    def grid_value(x, y):
        ix = int(math.floor(x))
        iy = int(math.floor(y))
        if 0 <= iy < grid.shape[0] and 0 <= ix < grid.shape[1]:
            return grid[iy, ix]
        return 0

    def distance_angle(x1, y1, x2, y2):
        return math.hypot(x2 - x1, y2 - y1), math.atan2(y2 - y1, x2 - x1)

    def nearest_node(node_list, x, y):
        return min(node_list, key=lambda node: math.hypot(node.x - x, node.y - y))

    def ant_random_point(goal, goal_sample_rate=0.05):
        if random.random() < goal_sample_rate:
            return goal
        visible_union = set()
        for pos in visited:
            key = str(pos)
            if key in visibility_map_raw:
                visible_union.update(tuple(cell) for cell in visibility_map_raw[key])
        if not visible_union:
            return random.randint(0, grid_size - 1), random.randint(0, grid_size - 1)
        candidates = list(visible_union)
        weights = [pheromone_grid[y, x] + 1e-6 for (x, y) in candidates]
        total = sum(weights)
        if total == 0 or not np.isfinite(total):
            return random.choice(candidates)
        probs = np.array(weights, dtype=np.float32)
        probs /= probs.sum()
        sampled_index = np.random.choice(len(candidates), p=probs)
        return candidates[sampled_index]

    def steer(from_node, to_x, to_y, step_size):
        dist, angle = distance_angle(from_node.x, from_node.y, to_x + 0.5, to_y + 0.5)
        step = min(step_size, dist)
        for i in range(1, int(step) + 1):
            nx = from_node.x + i * math.cos(angle)
            ny = from_node.y + i * math.sin(angle)
            if grid_value(nx, ny) == 0:
                return None
        return from_node.x + step * math.cos(angle), from_node.y + step * math.sin(angle)

    def interpolate_path(x1, y1, x2, y2):
        x1, y1 = int(math.floor(x1)), int(math.floor(y1))
        x2, y2 = int(math.floor(x2)), int(math.floor(y2))
        if grid[y1, x1] == 0 or grid[y2, x2] == 0:
            return []
        start, goal = (x1, y1), (x2, y2)
        visited_local = set()
        parent = {}
        cost = {start: 0}
        heap = [(0, start)]
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1),
                      (-1, -1), (-1, 1), (1, -1), (1, 1)]
        while heap:
            curr_cost, current = heapq.heappop(heap)
            if current in visited_local:
                continue
            visited_local.add(current)
            if current == goal:
                break
            for dx, dy in directions:
                nx, ny = current[0] + dx, current[1] + dy
                if 0 <= nx < grid.shape[1] and 0 <= ny < grid.shape[0] and grid[ny, nx] != 0:
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
        return path[::-1]

    def collision(x1, y1, x2, y2):
        return len(interpolate_path(x1, y1, x2, y2)) == 0

    def update_visibility(x, y, max_distance=grid_size):
        for angle in np.linspace(0, 2 * np.pi, 360, endpoint=False):
            for dist in range(1, max_distance):
                dx = x + dist * math.cos(angle)
                dy = y + dist * math.sin(angle)
                ix, iy = int(math.floor(dx)), int(math.floor(dy))
                if 0 <= ix < grid.shape[1] and 0 <= iy < grid.shape[0]:
                    visible_grid[iy, ix] = 1
                    if grid[iy, ix] == 0:
                        break
                else:
                    break

    def rrt(grid, start, goal, step_size=1, max_iter=5000):
        global visible_grid, pheromone_grid, visited
        start_node = Node(*start)
        goal_node = Node(*goal)
        update_visibility(start_node.x, start_node.y)
        nodes = [start_node]
        visited = set([(int(start_node.x), int(start_node.y))])
        for _ in range(max_iter):
            rand_x, rand_y = ant_random_point(goal)
            nearest = nearest_node(nodes, rand_x + 0.5, rand_y + 0.5)
            steered = steer(nearest, rand_x, rand_y, step_size)
            if steered:
                new_x, new_y = steered
                ix, iy = int(math.floor(new_x)), int(math.floor(new_y))
                if (ix, iy) not in visited and not collision(nearest.x, nearest.y, new_x, new_y):
                    new_node = Node(ix, iy)
                    new_node.x, new_node.y = new_x, new_y
                    new_node.parent = nearest
                    nodes.append(new_node)
                    visited.add((ix, iy))
                    update_visibility(new_x, new_y)
                    pheromone_grid *= 0.998
                    if abs(new_x - goal_node.x) + abs(new_y - goal_node.y) <= 3:
                        goal_node.parent = new_node
                        nodes.append(goal_node)
                        return nodes, len(nodes), True
        return [], 0, False

    if grid[start[1], start[0]] == 0 or grid[goal[1], goal[0]] == 0:
        return 0, 0.0, False

    start_time = time.time()
    nodes, steps, success = rrt(grid, start, goal, step_size=1, max_iter=5000)
    end_time = time.time()

    return steps, end_time - start_time, success