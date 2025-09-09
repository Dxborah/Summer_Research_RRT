import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib
import os
import numpy as np
import cv2
import networkx as nx
import numpy as np
import matplotlib.colors as mcolors
from matplotlib.collections import LineCollection

def grid_plot_rrt(grid, paths=None, line_color="orange", line_width=2):
    height, width = grid.shape

    # Define colormap
    cmap = mcolors.ListedColormap(
        ['blue', 'black', 'white', 'gray', 'red', 'green', 'blue', 'purple']
    )
    bounds = [-2.5, -1.5, -0.5, 0.5, 1.5, 2.5, 3.5, 4.5]
    norm = mcolors.BoundaryNorm(bounds, cmap.N)

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(grid, cmap=cmap, norm=norm, interpolation="nearest")

    # Colorbar
    cbar = plt.colorbar(ax.images[0], ticks=[-2, -1, 0, 1, 2, 3, 4], ax=ax)
    cbar.set_label("Grid States")
    cbar.set_ticklabels([
        'Edge of Coverage', 'Obstacles', 'Unseen', 'Free Seen Space',
        'Agent Position', 'Deployment Point', 'Target Point'
    ])

    ax.set_title("Partially Observable State")
    ax.set_xlabel("X-Coordinate")
    ax.set_ylabel("Y-Coordinate")
    ax.set_xticks(np.arange(0, width, step=5))
    ax.set_yticks(np.arange(0, height, step=5))
    ax.grid(False)

    # ---- Draw paths ----
    if paths:
        segs = []
        for path in paths:
            for (x1, y1), (x2, y2) in zip(path[:-1], path[1:]):
                # shift by 0.5 to go through cell centers
                segs.append([(x1 + 0.5, y1 - 0.5), (x2 + 0.5, y2 - 0.5)])

        lc = LineCollection(segs, colors=line_color, linewidths=line_width, zorder=5)
        ax.add_collection(lc)

    plt.tight_layout()
    plt.show()

def grid_plot(grid, width = 0,height = 0):
    
    width = grid.shape[1]
    height = grid.shape[0]

    plt.figure(figsize=(10, 10))
    plt.imshow(grid, cmap='jet', interpolation='nearest')
    cbar = plt.colorbar(ticks=[-2, -1, 0, 1, 3, 5, 10], label='Grid States')

    # Set custom tick labels
    cbar.set_ticklabels(['edge seen' ,'Blocked', 'Not Seen', 'Not Blocked', 'Agent Position', 'Deployment Point', 'Corner Goal', 'Agent Position'])

    plt.title('Partially Observable State')
    plt.xlabel('Width')
    plt.ylabel('Height')
    plt.xticks(ticks=np.arange(0, width, step=1))
    plt.yticks(ticks=np.arange(0, height, step=1))
    plt.grid(False)
    plt.show()

# def grid_plot_paper(grid):
#     # Save positions of green (3) cells before modifying
#     green_y, green_x = np.where(grid == 3)

#     # Set all green (3) cells to 0
#     grid[grid == 3] = 0

#     # Define custom colormap
#     cmap = mcolors.ListedColormap(['blue', 'black', 'gray', 'white', '#FF9800', '#87CEEB', 'blue', 'purple'])
#     bounds = [-2.5, -1.5, -0.5, 0.5, 1.5, 2.5, 3.5, 4.5]
#     norm = mcolors.BoundaryNorm(bounds, cmap.N)

#     plt.figure(figsize=(10, 10))
#     plt.imshow(grid, cmap=cmap, norm=norm)

#     # Overlay scatter markers
#     red_y, red_x = np.where(grid == 2)
#     purple_y, purple_x = np.where(grid == 5)

#     plt.scatter(red_x, red_y, s=400, facecolors='none', edgecolors='#FF9800', linewidths=2)
#     plt.scatter(purple_x, purple_y, s=400, facecolors='none', edgecolors='purple', linewidths=2)
#     plt.scatter(green_x, green_y, s=400, facecolors='none', edgecolors='#87CEEB', linewidths=2, marker='*')  # star

#     # Clean up axis
#     plt.axis('off')
#     plt.tight_layout(pad=0)
#     plt.show()

def grid_plot_paper2(grid_2, grid,agent_positions,targets,deployment_point,og_edge_seen):
    # Make a copy of the grid to modify
    display_grid = grid.copy()

    # Assign value 6 to cells where grid != 0 and grid_2 == 0
    mismatch_mask = (grid_2 != 1) & (grid == 1)
    display_grid[mismatch_mask] = 6
    print(targets)
    for agent in agent_positions:
        display_grid[agent[1]][agent[0]] = 2
    for target in og_edge_seen:
        display_grid[target[1]][target[0]] = -2
    for target in targets:
        display_grid[target[1]][target[0]] = 5

    display_grid[deployment_point[1]][deployment_point[0]] = 3

    # Define custom colormap including light gray for value 6
    cmap = mcolors.ListedColormap([
        'blue',      # -2
        'black',     # -1
        'black',      # 0
        'white',     # 1
        'red',       # 2
        'green',     # 3
        'blue',      # 4
        '#FF9800',    # 5
        'lightgrey'  # 6 → cells from mismatch
    ])
    bounds = [-2.5, -1.5, -0.5, 0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5]
    norm = mcolors.BoundaryNorm(bounds, cmap.N)

    plt.figure(figsize=(10, 10))
    plt.imshow(display_grid, cmap=cmap, norm=norm)

    # Overlay larger markers for red (2), green (3), and purple (5) cells from the original grid
    red_y, red_x = np.where(display_grid == 2)
    green_y, green_x = np.where(display_grid == 3)
    purple_y, purple_x = np.where(display_grid == 5)

    plt.scatter(red_x, red_y, s=700, facecolors='none', edgecolors='red', linewidths=2)
    plt.scatter(green_x, green_y, s=700, facecolors='none', edgecolors='green', linewidths=2, marker='s')

    plt.scatter(purple_x, purple_y, s=700, facecolors='none', edgecolors='#FF9800', linewidths=2, marker='*')

    plt.axis('off')
    plt.tight_layout(pad=0)
    plt.show()

def grid_plot_paper(grid):
    # Define custom colormap
    cmap = mcolors.ListedColormap(['blue', 'black', 'gray', 'white', 'red', 'green', 'blue', 'purple'])
    bounds = [-2.5, -1.5, -0.5, 0.5, 1.5, 2.5, 3.5, 4.5]
    norm = mcolors.BoundaryNorm(bounds, cmap.N)

    plt.figure(figsize=(10, 10))
    plt.imshow(grid, cmap=cmap, norm=norm)

    # Overlay larger markers for red (2) and green (3) cells
    red_y, red_x = np.where(grid == 2)
    green_y, green_x = np.where(grid == 3)
    purple_y, purple_x = np.where(grid == 5)

    plt.scatter(red_x, red_y, s=400, facecolors='none', edgecolors='red', linewidths=2)
    plt.scatter(green_x, green_y, s=400, facecolors='none', edgecolors='green', linewidths=2)
    plt.scatter(purple_x, purple_y, s=400, facecolors='none', edgecolors='purple', linewidths=2)

    # Remove all axis elements
    plt.axis('off')
    plt.tight_layout(pad=0)
    plt.show()

def grid_plot_2(grid):
    width, height = grid.shape[1], grid.shape[0]

    # Define custom colormap
    cmap = mcolors.ListedColormap(['blue', 'black', 'white', 'gray', 'red', 'green', 'blue', 'purple'])
    bounds = [-2.5, -1.5, -0.5, 0.5, 1.5, 2.5, 3.5, 4.5]
    norm = mcolors.BoundaryNorm(bounds, cmap.N)

    plt.figure(figsize=(10, 10))
    plt.imshow(grid, cmap=cmap, norm=norm)

    # Add colorbar with better tick labels
    cbar = plt.colorbar(ticks=[-2, -1, 0, 1, 2, 3, 4], label='Grid States')
    cbar.set_ticklabels([
        'Edge of Coverage', 'Obstacles', 'Unseen', 'Free Seen Space',
        'Agent Position', 'Deployment Point', 'Target Point'
    ])

    plt.title('Partially Observable State')
    plt.xlabel('X-Coordinate')
    plt.ylabel('Y-Coordinate')

    plt.xticks(ticks=np.arange(0, width, step=5))  # Reduce ticks for better readability
    plt.yticks(ticks=np.arange(0, height, step=5))

    plt.grid(False)
    plt.show()

def grid_plot_4(grid):
    print("here")
    h, w = grid.shape

    # Custom colormap
    cmap = mcolors.ListedColormap(
        ['blue', 'black', 'white', 'gray', 'red', 'green', 'blue', 'purple']
    )
    bounds = [-2.5, -1.5, -0.5, 0.5, 1.5, 2.5, 3.5, 4.5]
    norm = mcolors.BoundaryNorm(bounds, cmap.N)

    fig, ax = plt.subplots(figsize=(10, 10))

    # --- Use pcolormesh so we can draw per-cell borders ---
    x = np.arange(w + 1)
    y = np.arange(h + 1)

    mesh = ax.pcolormesh(
        x, y, grid,
        cmap=cmap, norm=norm,
        edgecolors=edge_color, linewidth=edge_width,
        shading="flat"
    )

    # Match imshow-style origin
    ax.invert_yaxis()
    ax.set_aspect("equal")

    # Add colorbar
    cbar = fig.colorbar(mesh, ax=ax, ticks=[-2, -1, 0, 1, 2, 3, 4], label='Grid States')
    cbar.set_ticklabels([
        'Edge of Coverage', 'Obstacles', 'Unseen', 'Free Seen Space',
        'Agent Position', 'Deployment Point', 'Target Point'
    ])

    ax.set_title("Partially Observable State")
    ax.set_xlabel("X-Coordinate")
    ax.set_ylabel("Y-Coordinate")

    ax.set_xticks(np.arange(0, w+1, 5))
    ax.set_yticks(np.arange(0, h+1, 5))

    plt.tight_layout()
    plt.show()

def grid_plot_3(grid, filename="save.png"):
    width, height = grid.shape[1], grid.shape[0]

    # Define custom colormap
    cmap = mcolors.ListedColormap(['blue', 'black', 'white', 'gray', 'red', 'green', 'blue', 'purple'])
    bounds = [-2.5, -1.5, -0.5, 0.5, 1.5, 2.5, 3.5, 4.5]
    norm = mcolors.BoundaryNorm(bounds, cmap.N)

    plt.figure(figsize=(10, 10))
    plt.imshow(grid, cmap=cmap, norm=norm)

    # Add colorbar with better tick labels
    cbar = plt.colorbar(ticks=[-2, -1, 0, 1, 2, 3, 4], label='Grid States')
    cbar.set_ticklabels([
        'Edge of Coverage', 'Obstacles', 'Unseen', 'Free Seen Space',
        'Agent Position', 'Deployment Point', 'Target Point'
    ])

    plt.title('Partially Observable State')
    plt.xlabel('X-Coordinate')
    plt.ylabel('Y-Coordinate')

    plt.xticks(ticks=np.arange(0, width, step=5))  # Reduce ticks for better readability
    plt.yticks(ticks=np.arange(0, height, step=5))

    plt.grid(False)
    if filename:
        plt.savefig(filename, bbox_inches='tight')
    plt.show()

def create_video_from_matrices(matrix_list, values_list, num_agents, output_folder="videos",  output_file="output_video.mp4", fps=10, temp_dir = "temp_frames",):
    matplotlib.use('Agg')  # Use Agg backend for non-interactive rendering
    if len(matrix_list) != len(values_list):
        raise ValueError("The number of matrices must match the number of values.")

    # Temporary folder to save individual frames
    #temp_dir = "temp_frames"
    os.makedirs(temp_dir, exist_ok=True)

    frame_files = []

    for i, (matrix, value,num) in enumerate(zip(matrix_list, values_list, num_agents)):
        plt.figure(figsize=(8, 6))
        plt.imshow(matrix, cmap='viridis', aspect='auto')
        plt.colorbar()
        plt.title(f"{num} Agents covering: {value} Grid Points")
        plt.axis('off')

        # Save the frame as an image
        frame_path = os.path.join(temp_dir, f"frame_{i:04d}.png")
        plt.savefig(frame_path)
        plt.close()
        frame_files.append(frame_path)

    # Ensure the output directory exists
    os.makedirs(output_folder, exist_ok=True)
    #output_path = os.path.join(output_folder, output_file)
    output_path = os.path.join(output_file)

    # Determine the frame size
    first_frame = cv2.imread(frame_files[0])
    height, width, layers = first_frame.shape
    frame_size = (width, height)

    # Create the video writer
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, frame_size)

    # Write each frame to the video
    for frame_file in frame_files:
        frame = cv2.imread(frame_file)
        out.write(frame)

    out.release()

    # Cleanup temporary frames
    for frame_file in frame_files:
        os.remove(frame_file)
    os.rmdir(temp_dir)

    print(f"Video saved as {output_path}")

def plot_graph(G):

    # Extract the positions of the nodes (points)
    pos = {node: node for node in G.nodes()}
    
    # Draw the graph with node positions and edge connections
    plt.figure(figsize=(8, 8))
    
    # Draw nodes (points)
    nx.draw_networkx_nodes(G, pos, node_size=50, node_color='blue', alpha=0.6)
    
    # Draw edges (connections between points)
    nx.draw_networkx_edges(G, pos, width=1.0, alpha=0.5, edge_color='black')
    
    # Labels for nodes (optional)
    nx.draw_networkx_labels(G, pos, font_size=10, font_color='white')
    
    # Display the graph
    plt.title("Graph of Points and Edges")
    plt.axis('off')  # Turn off the axis
    plt.show()

def visual_accuracy(new_vision,new_overlap,new_state,new_edge_seen,og_edge_seen,og_overlap,action_counter,graph_new,deployment_point,old_points):
    new_points = []
    modified_state = np.copy(new_state)
    for view in new_vision:
        vpath = find_shortest_path(graph_new, (view[1],view[0]), (deployment_point[1],deployment_point[0]),0)
        if vpath == None:
            modified_state[view[1],view[0]] = 0
    if action_counter == 1:
        tmp_edge_seen = set([])
        tmp_overlap = set([])
    else:
        tmp_edge_seen = og_edge_seen
        tmp_overlap = og_overlap

    for view in new_edge_seen - tmp_edge_seen:
        square_points = [(0,1),(0,-1),(1,0),(-1,0)]
        found = False
        for sp in square_points:
            new_view = (view[0] + sp[0], view[1] + sp[1])
            if modified_state[new_view[1]][new_view[0]] == 1:
                found = True
        if found == False:
            modified_state[view[1],view[0]] = 0
            new_points.append(view)

    for view in new_overlap - tmp_overlap:
        square_points = [(0,1),(0,-1),(1,0),(-1,0)]
        found = False
        for sp in square_points:
            new_view = (view[0] + sp[0], view[1] + sp[1])
            if modified_state[new_view[1]][new_view[0]] == 1:
                found = True
        if found == False:
            modified_state[view[1],view[0]] = 0
            new_points.append(view)

    for view in old_points:
        if modified_state[view[1]][view[0]] != 1:
            square_points = [(0,1),(0,-1),(1,0),(-1,0)]
            found = False
            for sp in square_points:
                new_view = (view[0] + sp[0], view[1] + sp[1])
                if modified_state[new_view[1]][new_view[0]] == 1:
                    found = True
            if found == False:
                modified_state[view[1],view[0]] = 0
                new_points.append(view)
    old_points = new_points

    return new_points, modified_state

def prep_frame(agents_goal,deployment_point,new_state,de_allocated_agents,agent_positions):
    modified_state = np.copy(new_state)
    for agent in agents_goal:
        if agent != deployment_point:
            modified_state[agent[1],agent[0]] = 20

    for agent in de_allocated_agents:
        if agent != deployment_point:
            modified_state[agent[1],agent[0]] = 15

    for agent in agent_positions:
        if agent != deployment_point:
            modified_state[agent[1],agent[0]] = 25
    modified_state[deployment_point[1],deployment_point[0]] = 30

    return modified_state