import numpy as np
import heapq
import random
import math
import json
from PIL import Image, ImageDraw

# === Load World Image and Visibility JSON ===
world_img = Image.open("IMG_8864.png").convert("RGB")
world_array = np.array(world_img)
grid_height, grid_width = world_array.shape[:2]

with open("FILE_3036.json", "r") as f:
    visibility_dict = json.load(f)

# === Build Visibility Set ===
visible_set = set()
for seen_coords in visibility_dict.values():
    for coord in seen_coords:
        visible_set.add(tuple(coord))

# === Convert image to binary obstacle grid ===
grid = np.ones((grid_height, grid_width), dtype=np.uint8) * 255
for y in range(grid_height):
    for x in range(grid_width):
        pixel = world_array[y, x]
        if np.array_equal(pixel, [0, 0, 0]):  # Black pixels = obstacles
            grid[y, x] = 0

# === RRT Classes and Helpers ===
class Node:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.parent = None

def distance_angle(x1, y1, x2, y2):
    distance = math.hypot(x2 - x1, y2 - y1)
    angle = math.atan2(y2 - y1, x2 - x1)
    return distance, angle

def nearest_node(node_list, x, y):
    return min(node_list, key=lambda node: math.hypot(node.x - x, node.y - y))

def steer(from_node, to_x, to_y, step_size):
    distance, angle = distance_angle(from_node.x, from_node.y, to_x, to_y)
    step = min(step_size, distance)

    for i in range(1, int(step) + 1):
        new_x = int(round(from_node.x + i * math.cos(angle)))
        new_y = int(round(from_node.y + i * math.sin(angle)))

        if not (0 <= new_x < grid.shape[1] and 0 <= new_y < grid.shape[0]):
            return None
        if grid[new_y, new_x] == 0:
            return None
        if i == int(step):
            return (new_x, new_y)
    return None

def ant_random_point(goal, goal_sample_rate=0.05):
    if random.random() < goal_sample_rate:
        return goal
    candidates = list(visible_set)
    if not candidates:
        return random.randint(0, grid.shape[1]-1), random.randint(0, grid.shape[0]-1)
    return random.choice(candidates)

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

def extract_path(goal_node):
    path = []
    node = goal_node
    while node is not None:
        path.append((node.x, node.y))
        node = node.parent
    return path[::-1]

# === RRT Main ===
def rrt(grid, start, goal, step_size=5, max_iter=2000):
    visited = set()
    start_node = Node(*start)
    goal_node = Node(*goal)
    nodes = [start_node]
    visited.add((start_node.x, start_node.y))

    # Debug: ensure start and goal are visible
    if start not in visible_set:
        visible_set.add(start)
    if goal not in visible_set:
        visible_set.add(goal)

    for _ in range(max_iter):
        # === Relax visibility constraints here ===
        if len(visible_set) < 100:  # Not enough data
            rand_x = random.randint(0, grid.shape[1] - 1)
            rand_y = random.randint(0, grid.shape[0] - 1)
        else:
            rand_x, rand_y = ant_random_point(goal)

        nearest = nearest_node(nodes, rand_x, rand_y)
        steered = steer(nearest, rand_x, rand_y, step_size)

        if steered:
            new_x, new_y = steered
            if (new_x, new_y) not in visited and not collision(nearest.x, nearest.y, new_x, new_y, grid):
                new_node = Node(new_x, new_y)
                new_node.parent = nearest
                nodes.append(new_node)
                visited.add((new_x, new_y))

                # Success condition (within 2-pixel box)
                if abs(new_x - goal[0]) + abs(new_y - goal[1]) <= 2:
                    goal_node.parent = new_node
                    nodes.append(goal_node)
                    print("✅ Path found!")
                    return nodes

    print("❌ Path not found")
    return None

# === Run RRT ===
start = (5, 5)
goal = (90, 90)
nodes = rrt(grid, start, goal)

# === Reveal Visible Areas ===
revealed_img = Image.new("RGB", world_img.size, (0, 0, 0))
draw = ImageDraw.Draw(revealed_img)
for coord in visible_set:
    x, y = coord
    if 0 <= x < grid_width and 0 <= y < grid_height:
        color = world_img.getpixel((x, y))
        draw.point((x, y), fill=color)

# === Draw Path on Top ===
if nodes:
    final_path = extract_path(nodes[-1])
    for (x, y) in final_path:
        draw.point((x, y), fill=(0, 255, 255))  # cyan path

    revealed_img.show()
# revealed_img.save("rrt_seen_world.png")
