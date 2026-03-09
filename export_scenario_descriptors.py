#!/usr/bin/env python
import os
import math
import numpy as np
import pandas as pd

# Import the scenario definitions
from config.RBM_define import params as RBM_params
from config.RDM_define import params as RDM_params


# ----------------------------------------------------------------------
# 1) Height grid extraction
# ----------------------------------------------------------------------
def extract_height_grid(city):
    """
    Try to obtain the 2-D height grid H(x,y) inside the CityConfigurator.

    Primary path:
        - Use city.height_grid_map if present (this is what your maps expose).

    Fallback:
        - Scan all 2-D numpy arrays on the city object and pick the one
          with the largest max height (most likely the building map).
    """
    # ---- primary: explicit attribute ---------------------------------
    if hasattr(city, "height_grid_map"):
        H = getattr(city, "height_grid_map")
        if isinstance(H, np.ndarray) and H.ndim == 2:
            H = H.astype(float)
            print(
                f"[export_scenario] Using city.height_grid_map as height grid; "
                f"shape={H.shape}, max_height={float(np.max(H)):.2f}"
            )
            return H

    # ---- fallback: scan 2‑D arrays -----------------------------------
    candidates = []
    for name in dir(city):
        try:
            val = getattr(city, name)
        except AttributeError:
            continue
        if isinstance(val, np.ndarray) and val.ndim == 2:
            candidates.append((name, val))

    if not candidates:
        raise RuntimeError(
            "[export_scenario] Could not find any 2‑D numpy arrays on city "
            "object to use as a height grid."
        )

    name, grid = max(candidates, key=lambda kv: float(np.max(kv[1])))
    grid = grid.astype(float)
    print(
        f"[export_scenario] Fallback: using city.{name} as height grid; "
        f"shape={grid.shape}, max_height={float(np.max(grid)):.2f}"
    )
    return grid


# ----------------------------------------------------------------------
# 2) Basic geometry descriptors
# ----------------------------------------------------------------------
def compute_obstacle_ratio(height_grid, grid_size_m):
    """
    Obstacle ratio = obstacle_area / total_area.

    We treat any cell with height > 0 as obstacle.
    """
    mask_obst = height_grid > 0.0
    num_cells = height_grid.size
    num_obst = int(mask_obst.sum())
    total_area_m2 = num_cells * (grid_size_m ** 2)
    obst_area_m2 = num_obst * (grid_size_m ** 2)
    ratio = obst_area_m2 / total_area_m2 if total_area_m2 > 0 else math.nan
    return ratio, mask_obst


def bresenham_line(x0, y0, x1, y1):
    """
    Bresenham line in grid coordinates, inclusive of endpoints.

    Returns a list of (x, y) integer coordinates along the line.
    """
    points = []
    dx = abs(x1 - x0)
    sx = 1 if x0 < x1 else -1
    dy = -abs(y1 - y0)
    sy = 1 if y0 < y1 else -1
    err = dx + dy
    x, y = x0, y0

    while True:
        points.append((x, y))
        if x == x1 and y == y1:
            break
        e2 = 2 * err
        if e2 >= dy:
            err += dy
            x += sx
        if e2 <= dx:
            err += dx
            y += sy

    return points


def estimate_los_blocked_ratio(mask_obst, n_pairs=1000, seed=0):
    """
    Approximate the fraction of straight-line paths between random free cells
    that are *blocked* by at least one obstacle cell.

    We only count intersections on interior cells (exclude endpoints).
    """
    rng = np.random.default_rng(seed)

    free_yx = np.argwhere(~mask_obst)  # indices of free cells
    if free_yx.shape[0] < 2:
        return math.nan

    H, W = mask_obst.shape
    blocked = 0
    total = 0

    for _ in range(n_pairs):
        idx1, idx2 = rng.integers(0, free_yx.shape[0], size=2)
        (y0, x0), (y1, x1) = free_yx[idx1], free_yx[idx2]

        pts = bresenham_line(int(x0), int(y0), int(x1), int(y1))
        # ignore endpoints; check interior only
        intersect = any(
            mask_obst[py, px]
            for (px, py) in pts[1:-1]
            if 0 <= px < W and 0 <= py < H
        )

        blocked += int(intersect)
        total += 1

    return blocked / total if total > 0 else math.nan


def compute_anchor_density(params, area_m2):
    """
    Anchor density = #known_anchors per km^2.
    """
    known_idx = params.get("known_device_idx", None)
    n_anchors = int(len(known_idx)) if known_idx is not None else 0
    area_km2 = area_m2 / 1e6
    if area_km2 <= 0:
        return math.nan
    return n_anchors / area_km2


# ----------------------------------------------------------------------
# 3) Per‑scenario descriptor
# ----------------------------------------------------------------------
def compute_descriptors_for_params(map_name, params):
    """
    Compute the scenario descriptors for a single map (RBM or RDM).
    """
    city = params["city"]
    urban = city.urban_config

    map_x = float(urban.map_x_len)
    map_y = float(urban.map_y_len)
    grid_size = float(urban.map_grid_size)

    area_m2 = map_x * map_y

    # Height grid and obstacle ratio
    Hgrid = extract_height_grid(city)
    obst_ratio, mask_obst = compute_obstacle_ratio(Hgrid, grid_size)

    # LoS blocked ratio (approximate)
    los_blocked_ratio = estimate_los_blocked_ratio(mask_obst, n_pairs=1000, seed=0)

    # Anchor density
    anchor_density = compute_anchor_density(params, area_m2)

    desc = {
        "map": map_name,
        "area_m2": area_m2,
        "obstacle_ratio": obst_ratio,
        "los_blocked_ratio": los_blocked_ratio,
        "anchor_density": anchor_density,
    }
    return desc


# ----------------------------------------------------------------------
# 4) Main
# ----------------------------------------------------------------------
def main():
    scenarios = {
        "RBM": RBM_params,
        "RDM": RDM_params,
    }

    rows = []
    for map_name, p in scenarios.items():
        print(f"\n[export_scenario] Processing map: {map_name}")
        desc = compute_descriptors_for_params(map_name, p)
        rows.append(desc)
        print(
            f"  area={desc['area_m2']:.1f} m^2, "
            f"obstacle_ratio={desc['obstacle_ratio']:.3f}, "
            f"los_blocked_ratio={desc['los_blocked_ratio']:.3f}, "
            f"anchor_density={desc['anchor_density']:.3f} anchors/km^2"
        )

    df = pd.DataFrame(rows)
    os.makedirs("result", exist_ok=True)
    out_csv = os.path.join("result", "scenario_descriptors.csv")
    df.to_csv(out_csv, index=False)
    print("\n[export_scenario] Saved scenario descriptors to:", out_csv)


if __name__ == "__main__":
    main()
