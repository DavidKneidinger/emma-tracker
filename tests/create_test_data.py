import numpy as np
import xarray as xr
import os
from datetime import datetime, timedelta


def linear_interpolate(t, t_start, t_end, start_val, end_val):
    if t <= t_start:
        return start_val
    if t >= t_end:
        return end_val
    frac = (t - t_start) / (t_end - t_start)
    return start_val + frac * (end_val - start_val)


def create_circular_cell(
    pr_array, center_y, center_x, radius, heavy_prec, moderate_prec
):
    """
    Draw a circular precipitation cell with a gradient from heavy_prec at the center to moderate_prec at the edge.
    """
    ny, nx = pr_array.shape
    y_indices, x_indices = np.indices((ny, nx))
    dist = np.sqrt((y_indices - center_y) ** 2 + (x_indices - center_x) ** 2)
    inside_mask = dist <= radius
    if np.any(inside_mask):
        pr_array[inside_mask] = moderate_prec + (heavy_prec - moderate_prec) * (
            1 - dist[inside_mask] / radius
        )


def save_single_timestep(
    out_dir, scenario_name, current_time, lat2d, lon2d, pr_slice, li_slice
):
    """
    Save a single timestep of test data as a netCDF file.
    """
    ds = xr.Dataset(
        {
            "pr": (("time", "y", "x"), pr_slice[np.newaxis, ...]),
            "li": (("time", "y", "x"), li_slice[np.newaxis, ...]),
        },
        coords={
            "time": [str(current_time)],
            "lat": (("y", "x"), lat2d),
            "lon": (("y", "x"), lon2d),
        },
        attrs={
            "description": "Complex scenario with growth, translation, merging, and splitting, plus a fast-mover.",
            "author": "David Kneidinger",
            "email": "david.kneidinger@uni-graz.at",
        },
    )
    ds["pr"].attrs["units"] = "mm/h"
    ds["pr"].attrs["long_name"] = "precipitation"
    ds["li"].attrs["units"] = "Kelvin"
    ds["li"].attrs["long_name"] = "lifting index"
    file_name = f"{scenario_name}_{current_time:%Y%m%d-%H-00}.nc_test"
    file_path = os.path.join(out_dir, file_name)
    ds.to_netcdf(file_path)


def create_test_data_scenario(out_dir):
    """
    Creates test data on a 200x200 grid:
      - The original three-cluster merge/split scenario, shifted up.
      - A large, non-growing system at the top which is in a non-convective LI band.
      - A new, isolated, fast-moving system at the bottom in a convective LI band.
    """
    os.makedirs(out_dir, exist_ok=True)

    # Grid (200 x 200)
    lat = np.arange(-10, 10, 0.1)
    lon = np.arange(0, 20, 0.1)
    lat2d, lon2d = np.meshgrid(lat, lon, indexing="ij")
    ny, nx = lat2d.shape

    # Time steps: 16 hourly steps
    start_time = datetime(2020, 1, 1, 0, 0)
    nt = 16

    # Precipitation parameters
    heavy_prec = 15.0
    moderate_prec = 5.0

    # --- Initial positions for systems ---
    top_init_y = 80
    mid_init_y = 100
    bot_init_y = 120
    init_x = 40

    non_convective_top_y = 170
    non_convective_top_radius = 15

    fast_mover_y = 30
    fast_mover_init_x = 20
    fast_mover_radius = 10

    # --- LI band definition ---
    # Everything at or above this y-index is non-convective.
    non_convective_y_threshold = 150

    def fast_mover_x_pos(t):
        return fast_mover_init_x + 21 * t

    def radius_func(t):
        if t <= 3:
            return linear_interpolate(t, 0, 3, 2, 7)
        elif t <= 13:
            return 7
        else:
            return linear_interpolate(t, 13, 15, 7, 2)

    def x_pos(t):
        return init_x + 5 * t

    def bot_y_func(t):
        if t <= 5:
            return bot_init_y
        elif t == 6:
            return bot_init_y - 10
        elif t == 7:
            return mid_init_y + 5
        elif t <= 11:
            return mid_init_y + 5
        elif t == 12:
            return bot_init_y - 5
        else:
            return bot_init_y

    def top_y_func(t):
        if t <= 7:
            return top_init_y
        elif t == 8:
            return mid_init_y - 15
        elif t <= 10:
            return mid_init_y - 15
        elif t == 11:
            return top_init_y
        else:
            return top_init_y - 5

    def mid_y_func(t):
        return mid_init_y

    scenario_name = "MCS-test"

    for t in range(nt):
        current_time = start_time + timedelta(hours=t)
        pr_slice = np.zeros((ny, nx), dtype=np.float32)
        li_slice = np.zeros((ny, nx), dtype=np.float32)

        # --- Create Lifting Index Bands ---
        li_slice[:non_convective_y_threshold, :] = -10.0  # Convective environment
        li_slice[non_convective_y_threshold:, :] = 0.0  # Non-convective environment

        # --- Create the three-cluster merging/splitting scenario ---
        r = radius_func(t)
        tx = int(x_pos(t))
        ty_top = int(top_y_func(t))
        ty_mid = int(mid_y_func(t))
        ty_bot = int(bot_y_func(t))

        create_circular_cell(pr_slice, ty_top, tx, r, heavy_prec, moderate_prec)
        create_circular_cell(pr_slice, ty_mid, tx, r + 2, heavy_prec, moderate_prec)
        create_circular_cell(pr_slice, ty_bot, tx, r, heavy_prec, moderate_prec)

        # --- Create the large, non-growing system in the non-convective band ---
        top_system_x = int(init_x + 4 * t)
        create_circular_cell(
            pr_slice,
            non_convective_top_y,
            top_system_x,
            non_convective_top_radius,
            heavy_prec,
            moderate_prec,
        )

        # --- Add the fast-moving system in the convective band ---
        fast_x = int(fast_mover_x_pos(t))
        if fast_x < nx - fast_mover_radius:
            create_circular_cell(
                pr_slice,
                fast_mover_y,
                fast_x,
                fast_mover_radius,
                heavy_prec,
                moderate_prec,
            )

        save_single_timestep(
            out_dir, scenario_name, current_time, lat2d, lon2d, pr_slice, li_slice
        )

    print("Updated test data created successfully at", out_dir)


if __name__ == "__main__":
    create_test_data_scenario("./Test/data/")
