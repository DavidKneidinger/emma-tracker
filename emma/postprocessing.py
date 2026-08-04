"""
emma/postprocessing.py

Physics-based Filtering and Post-Processing for MCS Tracking.

This module implements the final stage of the tracking pipeline. It refines the 
raw tracking results by:
1.  **Extracting** physical properties (Area, Main Var, Env Var) for every track timestep.
2.  **Aggregating** these properties to calculate lifetime statistics and kinematics.
3.  **Filtering** tracks based on physics-based thresholds (e.g., instability, straightness, area volatility).
4.  **Consolidating** results into global CSV files.
5.  **Restoring** Clean NetCDF files containing only the valid, accepted MCS tracks.
"""

import os
import glob
import logging
import concurrent.futures
import numpy as np
import pandas as pd
import xarray as xr
from scipy.stats import skew
from typing import List
import operator

# Import project-specific helpers
from .input_output import load_env_var_data, load_main_var_data
from .postprocessing_helper_func import (
    prepare_grid_dict,
    calculate_grid_area_map,
    calculate_kinematics,
    calculate_area_change,
)

logger = logging.getLogger(__name__)

OP_MAP = {
    ">=": operator.ge,
    "<=": operator.le,
    ">": operator.gt,
    "<": operator.lt,
}


def update_global_csv(new_df: pd.DataFrame, file_path: str, time_col: str, year: int):
    """
    Updates a global CSV file by appending new data and replacing existing data for the given year.

    This ensures the global file remains "one for all" without duplicating rows
    if a specific year is re-processed.

    Args:
        new_df (pd.DataFrame): The DataFrame containing data for the current processing year.
        file_path (str): The full path to the global CSV file.
        time_col (str): The name of the datetime column used to identify the year (e.g., 'datetime' or 'start_time').
        year (int): The specific year being processed.
    """
    if new_df.empty:
        return

    # Ensure time column is datetime objects
    new_df[time_col] = pd.to_datetime(new_df[time_col])

    if os.path.exists(file_path):
        # Read existing global data
        existing_df = pd.read_csv(file_path)
        existing_df[time_col] = pd.to_datetime(existing_df[time_col])

        # Remove old entries for this specific year (Clean Overwrite)
        existing_df = existing_df[existing_df[time_col].dt.year != year]

        # Concatenate old data with new data
        updated_df = pd.concat([existing_df, new_df], ignore_index=True)
    else:
        updated_df = new_df

    # Sort by time and track number for tidiness
    if "track_number" in updated_df.columns:
        updated_df = updated_df.sort_values(by=[time_col, "track_number"])
    else:
        updated_df = updated_df.sort_values(by=[time_col])

    # Save back to CSV
    updated_df.to_csv(file_path, index=False)
    logger.info(f"Updated global record: {file_path}")


def process_single_timestep(
    file_path: str,
    env_files: List[str],
    main_files: List[str],
    config: object,
    main_var_name: str,
    env_var_name: str,
    lat_name: str,
    lon_name: str,
) -> List[dict]:
    """
    Worker function to extract physical properties for all tracks in a single NetCDF file.

    Args:
        file_path (str): Path to the raw tracking NetCDF file.
        env_files (List[str]): List of available Environmental Variable file paths.
        main_files (List[str]): List of available Main Variable file paths.
        config (object): Configuration object containing variable names and thresholds.
        main_var_name (str): Variable name of the main tracking field in NetCDF files.
        env_var_name (str): Variable name of the environmental field in NetCDF files.
        lat_name (str): Name of the latitude coordinate.
        lon_name (str): Name of the longitude coordinate.

    Returns:
        List[dict]: A list of dictionaries, where each dictionary contains the extracted
                    properties for a single track at this timestep. Returns an empty list
                    if no tracks are found or errors occur.
    """
    os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
    results = []

    with xr.open_dataset(file_path, engine="netcdf4") as ds:
        # Check for tracks
        if "active_track_id" not in ds:
            return []

        active_ids = ds["active_track_id"].values
        if len(active_ids) == 0:
            return []

        time_val = ds["time"].values[0]
        time_str = str(time_val)

        ds = ds.isel(time=0)
        # --- 1. Grid & Area Calculation ---
        # Extract grid coordinates robustly (handling 1D/2D and different var names)
        grid_dict = prepare_grid_dict(ds)

        # Calculate cell areas (km2) handling Regular vs Irregular grids
        area_map_km2 = calculate_grid_area_map(grid_dict)

        # --- 2. Environmental Data Loading ---
        t_pd = pd.to_datetime(time_val)
        time_key_exact = t_pd.strftime("%Y%m%dT%H%M")
        time_key_hour = t_pd.strftime("%Y%m%dT%H")

        # Load Environmental Variable (matches exact minute key first, falls back to hour key)
        env_file = next(
            (f for f in env_files if time_key_exact in os.path.basename(f)),
            next((f for f in env_files if time_key_hour in os.path.basename(f)), None),
        )

        current_env_var = None  # Initialize as None
        if env_file:
            current_env_var = load_env_var_data(
                env_file, env_var_name, lat_name, lon_name
            )[-1].squeeze()

        # Load Main Variable (matches exact minute key first, falls back to hour key)
        main_file = next(
            (f for f in main_files if time_key_exact in os.path.basename(f)),
            next((f for f in main_files if time_key_hour in os.path.basename(f)), None),
        )

        current_main_var = None
        if main_file:
            current_main_var = load_main_var_data(
                main_file, main_var_name, lat_name, lon_name
            )[-1].squeeze()

        # Skip detailed physics if environmental data is missing
        if current_env_var is None or current_main_var is None:
            logger.warning(f"Skipping physics for {time_str} (missing env data)")
            return []

        # --- 3. Property Extraction per Track ---
        mcs_map = ds["mcs_id"].values

        for track_id in active_ids:
            mask = mcs_map == track_id
            if not np.any(mask):
                continue

            # Centroid
            idx = np.where(ds["active_track_id"].values == track_id)[0][0]
            center_lat = ds["active_track_lat"].values[idx]
            center_lon = ds["active_track_lon"].values[idx]

            # Geometric Properties
            track_area = np.sum(area_map_km2[mask])

            # Physical Properties
            mean_env_var = np.nan
            if current_env_var is not None and current_env_var.shape == mask.shape:
                mean_env_var = np.nanmean(current_env_var.values[mask])

            p_vals = np.array([])
            if current_main_var is not None and current_main_var.shape == mask.shape:
                p_vals = current_main_var.values[mask]

            mean_main_var = np.nanmean(p_vals) if len(p_vals) > 0 else np.nan
            max_main_var = np.nanmax(p_vals) if len(p_vals) > 0 else np.nan

            main_var_skew = np.nan

            if len(p_vals) > 5:
                main_var_skew = skew(p_vals, nan_policy="omit")

            # Convective / Stratiform Partitioning
            detection_parameters = config.detection_parameters
            conv_thresh = detection_parameters.core_threshold
            main_op = getattr(detection_parameters, "main_var_operator", ">=")

            conv_op_func = OP_MAP.get(main_op, operator.ge)
            conv_mask = conv_op_func(p_vals, conv_thresh)

            conv_area = np.sum(area_map_km2[mask][conv_mask])
            strat_area = track_area - conv_area

            results.append(
                {
                    "track_number": f"{track_id}-{pd.to_datetime(time_val).year}",
                    "datetime": time_str,
                    "center_lat": center_lat,
                    "center_lon": center_lon,
                    "area_km2": track_area,
                    "mean_env_var": mean_env_var,
                    "mean_main_var": mean_main_var,
                    "max_main_var": max_main_var,
                    "main_var_skew": main_var_skew,
                    "convective_area_km2": conv_area,
                    "stratiform_area_km2": strat_area,
                }
            )

    return results


def restore_filtered_files(
    raw_files: List[str],
    valid_ids: set,
    output_dir: str,
    input_root_dir: str,
    config: object = None,
):
    """
    Generates the final NetCDF files containing only valid MCS tracks.
    Attaches active run metadata to output attributes.
    """
    active_meta = config.get_active_metadata() if config else {}

    for f_path in raw_files:
        relative_structure = os.path.relpath(f_path, input_root_dir)
        out_path = os.path.join(output_dir, relative_structure)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)

        with xr.open_dataset(f_path, engine="netcdf4") as ds:
            # Attach active metadata top-level global attributes
            for k, v in active_meta.items():
                ds.attrs[k] = v

            if "active_track_id" not in ds:
                ds.to_netcdf(out_path)
                continue

            raw_ids = ds["active_track_id"].values
            all_file_ids = np.atleast_1d(raw_ids)
            valid_tracks_in_file = [tid for tid in all_file_ids if tid in valid_ids]

            if not valid_tracks_in_file:
                vars_to_drop = [
                    "active_track_id",
                    "active_track_lat",
                    "active_track_lon",
                    "mcs_id",
                ]
                ds_filtered = ds.drop_vars(
                    [v for v in vars_to_drop if v in ds], errors="ignore"
                )

                if "final_labeled_regions" in ds_filtered:
                    ds_filtered["final_labeled_regions"].values[:] = 0

                ds_filtered.to_netcdf(out_path)
                continue

            track_dims = ds["active_track_id"].dims
            if len(track_dims) == 0:
                ds_filtered = ds.copy()
            else:
                dim_name = track_dims[0]
                valid_indices = np.where(np.isin(all_file_ids, valid_tracks_in_file))[0]
                ds_filtered = ds.isel({dim_name: valid_indices})

            if "final_labeled_regions" in ds_filtered:
                mask_da = ds_filtered["final_labeled_regions"]
                mask_vals = mask_da.values
                new_mask = np.where(
                    np.isin(mask_vals, valid_tracks_in_file), mask_vals, 0
                )
                ds_filtered["final_labeled_regions"].values[:] = new_mask

            ds_filtered.to_netcdf(out_path)


def run_postprocessing_year(
    year: int,
    raw_tracking_output_dir: str,
    tracking_output_dir: str,
    main_var_name: str,
    env_var_name: str,
    lat_name: str,
    lon_name: str,
    config: object,
):
    """
    Orchestrates the post-processing pipeline for a specific year.

    This function:
    1.  Extracts physical properties for the given year.
    2.  Aggregates properties into track summaries.
    3.  **Updates the global CSV files** located in the root post-processing directory.
    4.  Filters tracks based on configuration thresholds.
    5.  Restores valid tracks into clean NetCDF files (stored in the yearly folder).

    Args:
        year (int): The year being processed.
        raw_tracking_output_dir (str): Input directory containing raw tracking NetCDFs.
        tracking_output_dir (str): Output directory for the final NetCDFs (e.g., .../2020/).
        main_var_name (str): Variable name of the main tracking field.
        env_var_name (str): Variable name of the environmental field.
        lat_name (str): Name of the latitude coordinate.
        lon_name (str): Name of the longitude coordinate.
        config (object): Global configuration object.
    """
    logger.info(f"--- Starting Post-Processing for Year: {year} ---")
    year_data_dir = os.path.join(raw_tracking_output_dir, str(year))

    if not os.path.exists(year_data_dir):
        raise FileNotFoundError(
            f"Raw tracking directory for year {year} not found: {year_data_dir}"
        )

    # Recursive search to find files in subdirectories (e.g., 2020/08/*.nc)
    raw_files = sorted(
        glob.glob(os.path.join(year_data_dir, "**", "*.nc"), recursive=True)
    )

    if not raw_files:
        raise FileNotFoundError(f"No .nc files found in {year_data_dir}")

    logger.info(f"Found {len(raw_files)} raw tracking files in {year_data_dir}")

    # Ensure yearly output directory exists for NetCDFs
    os.makedirs(tracking_output_dir, exist_ok=True)

    # 2. Locate Environmental Data (Env Var and Main Var)
    env_dir = config.env_var_data_directory
    main_dir = config.main_var_data_directory

    all_env = sorted(glob.glob(os.path.join(env_dir, "**", "*.nc"), recursive=True))
    all_main = sorted(glob.glob(os.path.join(main_dir, "**", "*.nc"), recursive=True))

    # Filter files relevant for this year to optimize search
    env_files_year = [f for f in all_env if str(year) in os.path.basename(f)]
    main_files_year = [f for f in all_main if str(year) in os.path.basename(f)]

    # --- STEP 1: Extract Timestep Properties ---
    logger.info("STEP 1: Extracting timestep properties...")
    all_timestep_rows = []

    if config.use_multiprocessing and config.number_of_cores > 1:
        logger.info(
            f"Running extraction in PARALLEL mode ({config.number_of_cores} cores)..."
        )
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=config.number_of_cores
        ) as executor:
            futures = {
                executor.submit(
                    process_single_timestep,
                    f,
                    env_files_year,
                    main_files_year,
                    config,
                    main_var_name,
                    env_var_name,
                    lat_name,
                    lon_name,
                ): f
                for f in raw_files
            }

            for future in concurrent.futures.as_completed(futures):
                res = future.result()
                if res:
                    all_timestep_rows.extend(res)
    else:
        logger.info("Running extraction in SERIAL mode...")
        for f in raw_files:
            res = process_single_timestep(
                f,
                env_files_year,
                main_files_year,
                config,
                main_var_name,
                env_var_name,
                lat_name,
                lon_name,
            )
            if res:
                all_timestep_rows.extend(res)

    df_timesteps = pd.DataFrame(all_timestep_rows)
    df_timesteps["datetime"] = pd.to_datetime(df_timesteps["datetime"])
    df_timesteps = df_timesteps.sort_values(by=["datetime", "track_number"])

    # Update GLOBAL Timestep CSV
    csv_timestep_path = os.path.join(tracking_output_dir, "mcs_timestep_properties.csv")
    update_global_csv(df_timesteps, csv_timestep_path, "datetime", year)

    # --- STEP 2: Aggregate & Analyze ---
    logger.info("STEP 2: Analyzing track properties...")

    aggregations = {
        "datetime": ["min", "max", "count"],
        "area_km2": ["mean", "max"],
        "mean_env_var": ["mean"],
        "mean_main_var": ["mean"],
        "max_main_var": ["max"],
        "convective_area_km2": ["mean"],
        "stratiform_area_km2": ["mean"],
        "main_var_skew": ["mean"],
    }

    # GroupBy & Aggregation
    df_summary = df_timesteps.groupby("track_number").agg(aggregations)
    df_summary.columns = ["_".join(col).strip() for col in df_summary.columns.values]

    rename_map = {
        "datetime_min": "start_time",
        "datetime_max": "end_time",
        "datetime_count": "duration_steps",
        "area_km2_mean": "lifetime_mean_area_km2",
        "area_km2_max": "max_area_km2",
        "mean_env_var_mean": "lifetime_mean_env_var",
        "mean_main_var_mean": "lifetime_mean_main_var",
        "max_main_var_max": "peak_max_main_var",
    }
    df_summary = df_summary.rename(columns=rename_map)

    # Calculate Kinematics & Volatility
    kinematics = df_timesteps.groupby("track_number").apply(calculate_kinematics)
    volatility = df_timesteps.groupby("track_number").apply(calculate_area_change)

    df_summary = df_summary.merge(kinematics, on="track_number")
    df_summary = df_summary.merge(volatility, on="track_number")

    # Add track-number as column and put it first
    df_summary = df_summary.reset_index()
    df_summary.insert(0, "track_number", df_summary.pop("track_number"))

    # Update GLOBAL Summary CSV
    csv_summary_path = os.path.join(tracking_output_dir, "mcs_track_summary.csv")
    update_global_csv(df_summary, csv_summary_path, "start_time", year)

    # --- STEP 3: Filter ---
    logger.info("STEP 3: Filtering tracks...")

    filters = config.postprocessing_filters
    env_op = getattr(filters, "env_var_operator", "<=")
    thresh_env = filters.env_var_threshold
    thresh_straight = filters.track_straightness_threshold
    thresh_vol = filters.max_area_volatility

    logger.info(
        f"Filtering Criteria: Env Var {env_op} {thresh_env}, Straightness > {thresh_straight}, Volatility < {thresh_vol}"
    )

    # Filter Logic
    env_op_func = OP_MAP.get(env_op, operator.le)
    env_cond = env_op_func(df_summary["lifetime_mean_env_var"], thresh_env)

    accepted_mask = (
        env_cond
        & (df_summary["track_straightness"] > thresh_straight)
        & (df_summary["max_area_volatility"] < thresh_vol)
    )

    df_accepted = df_summary[accepted_mask]
    df_rejected = df_summary[~accepted_mask]

    # Update GLOBAL Accepted/Rejected CSVs
    update_global_csv(
        df_accepted,
        os.path.join(tracking_output_dir, "mcs_track_summary_ACCEPTED.csv"),
        "start_time",
        year,
    )
    update_global_csv(
        df_rejected,
        os.path.join(tracking_output_dir, "mcs_track_summary_REJECTED.csv"),
        "start_time",
        year,
    )

    valid_ids = set(df_accepted.index.tolist())
    logger.info(f"Accepted: {len(valid_ids)}, Rejected: {len(df_rejected)}")

    # --- STEP 4: Restore ---
    logger.info("STEP 4: Restoring filtered NetCDF files...")

    restore_filtered_files(
        raw_files, valid_ids, tracking_output_dir, raw_tracking_output_dir
    )
    logger.info(f"Restoration complete in {tracking_output_dir}")
