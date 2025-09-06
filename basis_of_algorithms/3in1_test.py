import numpy as np
from PIL import Image
from pathlib import Path
import shelve
import random

def find_270_corners(grid):
    rows, cols = grid.shape
    corner_positions = []
    for i in range(rows - 1):
        for j in range(cols - 1):
            block = grid[i:i+2, j:j+2]
            if block.sum() == 3:  # L-shape
                black_idx = np.argwhere(block == 0)[0]
                di, dj = 1 - black_idx[0], 1 - black_idx[1]
                corner_i, corner_j = i + di, j + dj
                corner_positions.append((corner_i, corner_j))
    return corner_positions

def find_shelf_files(shelves_dir, dungeon_prefix):
    path_obj = Path(shelves_dir)
    dat_files = list(path_obj.rglob(f"{dungeon_prefix}_*.shelf.dat"))
    shelf_bases = [str(dat_file)[:-4] for dat_file in dat_files]
    return sorted(shelf_bases)

def get_visibility_from_shelves(shelf_files):
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
    return merged_visibility

def _xy_to_rc(cells_xy, shape):
    """Convert iterable of (x,y) to (row,col)=(y,x), keep only in-bounds."""
    rows, cols = shape
    out = []
    for xy in cells_xy:
        # handle list/tuple/np types robustly
        x = int(xy[0]); y = int(xy[1])
        r, c = y, x
        if 0 <= r < rows and 0 <= c < cols:
            out.append((r, c))
    return out

def make_three_panel_image(world_png, shelves_dir, dungeon_prefix, output_png):
    # --- load world as 0/1 (1=walkable, 0=wall) ---
    world = np.array(Image.open(world_png).convert("L")) // 255
    rows, cols = world.shape

    # --- find shelf files & merge visibility (unchanged helpers you already have) ---
    shelf_files = find_shelf_files(shelves_dir, dungeon_prefix)
    merged_visibility = get_visibility_from_shelves(shelf_files)

    # --- find corners (your existing function) ---
    corners = find_270_corners(world)
    if not corners:
        print("No 270° corners found.")
        return

    # pick one random corner (row,col)
    corner_rc = random.choice(corners)

    # shelves use (x,y), so keys are (col,row)
    key_xy_str = f"({corner_rc[1]}, {corner_rc[0]})"
    raw_visible_xy = merged_visibility.get(key_xy_str, [])

    # convert cells (x,y) -> (row,col), keep in bounds
    visible_rc_all = set(_xy_to_rc(raw_visible_xy, world.shape))

    # keep **only walkable** cells to avoid painting walls
    visible_rc = {(r, c) for (r, c) in visible_rc_all if world[r, c] == 1}

    # --- Panel 1: original (white=walkable, black=wall) ---
    panel1 = np.stack([world * 255, world * 255, world * 255], axis=-1)
    # mark chosen corner in red
    pr, pc = corner_rc
    panel1[pr, pc] = [255, 0, 0]

    # --- Panel 2: visibility from the corner (green) ---
    panel2 = panel1.copy()
    for r, c in visible_rc:
        panel2[r, c] = [0, 255, 0]
    # keep the corner marker visible (red overwrites green at that cell)
    panel2[pr, pc] = [255, 0, 0]

    # --- Panel 3: edge of coverage (green = visible, blue = frontier of visibility) ---
    panel3 = np.stack([world * 255, world * 255, world * 255], axis=-1)

    # Step 1: paint all visible cells green
    for r, c in visible_rc:
        panel3[r, c] = [0, 255, 0]

    # Step 2: frontier = neighbors of visible cells that are walkable but not visible
    frontier_rc = set()
    for r, c in visible_rc:
        for dr, dc in [(-1,0), (1,0), (0,-1), (0,1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols:
                if world[nr, nc] == 1 and (nr, nc) not in visible_rc:
                    frontier_rc.add((nr, nc))

    # Step 3: paint frontier cells blue
    for r, c in frontier_rc:
        panel3[r, c] = [0, 0, 255]

    # Step 4: keep the chosen corner red
    panel3[pr, pc] = [255, 0, 0]


    # --- combine horizontally and save ---
    combined = np.concatenate([panel1, panel2, panel3], axis=1)
    Image.fromarray(combined.astype(np.uint8)).save(output_png)

# Example usage
make_three_panel_image(
    world_png="../divided_world_1/15000-15999/dungeon_2143.png",
    shelves_dir="../divided_world_1",
    dungeon_prefix="dungeon_2143",
    output_png="three_panel_example3.png"
)
