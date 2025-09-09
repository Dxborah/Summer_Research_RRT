import numpy as np
import networkx as nx
from itertools import chain
from matplotlib import pyplot as plt

def library_format(position):
    return '(' + str(position[0]) + ',' + str(position[1]) + ')'


def get_all_adjacent_points(points):
    return {(x + dx, y + dy) for x, y in points for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1))}


def dijkstra_shortest_path(G, start, end):
    try:
        path = nx.dijkstra_path(G, start, end)
        length = nx.dijkstra_path_length(G, start, end)
        return path, length
    except nx.NetworkXNoPath:
        return None, float('inf')  # No path exists
   
def flip_points(point):
    return (point[1],point[0])

def flip_points_back(point):
    return (point[1],point[0])

def assign_points(tar_grid, point_set, value):
    if point_set:
        coords = np.array(list(point_set))
        tar_grid[coords[:, 1], coords[:, 0]] = value
    return tar_grid

def process_vision_clean(agent_positions, grid, visibility_data, step= None, target=None):


    blocked_points= set(visibility_data["blocked"])
    not_seen = set(visibility_data["all"]).difference(blocked_points)

    vision_sets = set(chain.from_iterable([visibility_data[f'({i[0]}, {i[1]})'] 
                                           for i in (agent_positions 
                                                     if len(agent_positions) > 1 else agent_positions)])) 
        
    not_seen = not_seen.difference(vision_sets)
    
    adjacent_overlaps = get_all_adjacent_points(vision_sets)
    adjacent_seen = get_all_adjacent_points(not_seen)
    
    overlaps = adjacent_overlaps.intersection(blocked_points)

    edge_seen = adjacent_seen.intersection(vision_sets)
    
    
    grid= assign_points(grid, blocked_points, 0)
    grid= assign_points(grid, overlaps, -1)
    grid= assign_points(grid, not_seen, 0)
    grid= assign_points(grid, edge_seen, -2)

    paint_grid = np.copy(grid)

    if step is not None:
        
        paint_grid-= 2

        paint_grid= assign_points(paint_grid, agent_positions, 1)

        paint_grid= assign_points(paint_grid, agent_positions[-1:], 2)

        if target is not None:
            paint_grid= assign_points(paint_grid, [target], 0)

        plt.figure(figsize=(12, 10), dpi=300)
        plt.imshow(paint_grid, cmap='viridis', interpolation='nearest')
        plt.title(f'Step {step}')
        plt.colorbar()
        plt.savefig(f'grid_step_{step}.png', dpi=300, bbox_inches='tight')
        plt.close()



    return vision_sets, grid, edge_seen