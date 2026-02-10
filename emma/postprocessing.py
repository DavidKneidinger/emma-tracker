"""
emma/postprocessing.py

Physics-based Filtering and Post-Processing for MCS Tracking.

This module implements the final stage of the tracking pipeline. It refines the 
raw tracking results by:
1.  **Extracting** physical properties (Area, Precip, LI) for every track timestep.
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

# Import project-specific helpers
from .input_output import load_lifted_index_data, load_precipitation_data
from .postprocessing_helper_func import (
    prepare_grid_dict,
    calculate_grid_area_map, 
    calculate_kinematics, 
    calculate_area_change
)

logger = logging.getLogger(__name__)

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
    if 'track_number' in updated_df.columns:
        updated_df = updated_df.sort_values(by=[time_col, 'track_number'])
    else:
        updated_df = updated_df.sort_values(by=[time_col])

    # Save back to CSV
    updated_df.to_csv(file_path, index=False)
    logger.info(f"Updated global record: {file_path}")


def process_single_timestep(file_path: str, li_files: List[str], precip_files: List[str], config: object, 
                            precip_var_name: str, lifted_index_var_name: str, lat_name: str, lon_name: str) -> List[dict]:
    """
    Worker function to extract physical properties for all tracks in a single NetCDF file.

    Args:
        file_path (str): Path to the raw tracking NetCDF file.
        li_files (List[str]): List of available Lifted Index file paths.
        precip_files (List[str]): List of available Precipitation file paths.
        config (object): Configuration object containing variable names and thresholds.

    Returns:
        List[dict]: A list of dictionaries, where each dictionary contains the extracted 
                    properties for a single track at this timestep. Returns an empty list 
                    if no tracks are found or errors occur.
    """
    os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'
    results = []
    
    with xr.open_dataset(file_path, engine='netcdf4') as ds:
        # Check for tracks
        if 'active_track_id' not in ds:
            return []
        
        active_ids = ds['active_track_id'].values
        if len(active_ids) == 0:
            return []

        time_val = ds['time'].values[0]
        time_str = str(time_val)

        ds = ds.isel(time=0)
        # --- 1. Grid & Area Calculation ---
        # Extract grid coordinates robustly (handling 1D/2D and different var names)
        grid_dict = prepare_grid_dict(ds)
        
        # Calculate cell areas (km2) handling Regular vs Irregular grids
        area_map_km2 = calculate_grid_area_map(grid_dict)
        
        # --- 2. Environmental Data Loading ---
        t_pd = pd.to_datetime(time_val)
        time_key = t_pd.strftime("%Y%m") # Match monthly file pattern

        # Load Lifted Index (using shared helper)
        li_file = next((f for f in li_files if time_key in os.path.basename(f)), None)
        
        current_li = None  # Initialize as None
        if li_file:        # Only load if file exists
            current_li = load_lifted_index_data(li_file, lifted_index_var_name, lat_name, lon_name)[-1]
            
        # Load Precipitation 
        precip_file = next((f for f in precip_files if time_key in os.path.basename(f)), None)

        current_precip = None
        if precip_file:
            current_precip = load_precipitation_data(precip_file, precip_var_name, lat_name, lon_name)[-1]

        
        # Skip detailed physics if environmental data is missing
        if current_li is None or current_precip is None:
            logger.warning(f"Skipping physics for {time_str} (missing env data)")
            return []

        # --- 3. Property Extraction per Track ---
        mcs_map = ds['mcs_id'].values

        for track_id in active_ids:
            mask = (mcs_map == track_id)
            if not np.any(mask):
                continue
            
            # Centroid
            idx = np.where(ds['active_track_id'].values == track_id)[0][0]
            center_lat = ds['active_track_lat'].values[idx]
            center_lon = ds['active_track_lon'].values[idx]
            
            # Geometric Properties
            track_area = np.sum(area_map_km2[mask])
            
            # Physical Properties
            mean_li = np.nanmean(current_li.values[mask])
            p_vals = current_precip.values[mask]
            mean_precip = np.nanmean(p_vals)
            max_precip = np.nanmax(p_vals)
            
            precip_skew = np.nan
            if len(p_vals) > 5:
                precip_skew = skew(p_vals, nan_policy='omit')
            
            # Convective / Stratiform Partitioning
            detection_parameters = config.detection_parameters
            conv_thresh = detection_parameters.heavy_precip_threshold

            conv_mask = (p_vals > conv_thresh)
            conv_area = np.sum(area_map_km2[mask][conv_mask])
            strat_area = track_area - conv_area

            results.append({
                'track_number': int(track_id),
                'datetime': time_str,
                'center_lat': center_lat,
                'center_lon': center_lon,
                'area_km2': track_area,
                'mean_li': mean_li,
                'mean_precip': mean_precip,
                'max_precip': max_precip,
                'precip_skew': precip_skew,
                'convective_area_km2': conv_area,
                'stratiform_area_km2': strat_area
            })

    return results

def restore_filtered_files(raw_files: List[str], valid_ids: set, output_dir: str, input_root_dir: str):
    """
    Generates the final NetCDF files containing only valid MCS tracks.

    Reads the raw tracking files, masks out any track IDs that are not in the 
    `valid_ids` set, and saves the cleaned file to the output directory.

    Args:
        raw_files (List[str]): List of paths to raw tracking NetCDF files.
        valid_ids (set): Set of track IDs that passed the filtering criteria.
        output_dir (str): Directory where the final NetCDF files will be saved.
    """
    os.makedirs(output_dir, exist_ok=True)

    for f_path in raw_files:
        with xr.open_dataset(f_path, engine='netcdf4') as ds:
            # 1. Handle files with NO tracks (e.g. empty timesteps)
            if 'active_track_id' not in ds:
                continue
            
            # 2. Safely get IDs (Handling Scalar vs Array)
            #    We use .values to get numpy array, then atleast_1d to ensure iteration works
            raw_ids = ds['active_track_id'].values
            all_file_ids = np.atleast_1d(raw_ids)
            
            # 3. Identify which tracks in THIS file are valid
            #    (valid_ids contains ALL valid tracks for the whole year)
            valid_tracks_in_file = [tid for tid in all_file_ids if tid in valid_ids]
            
            # If the file contains tracks, but NONE are valid, we skip saving it.
            if not valid_tracks_in_file:
                continue

            # 4. Filter the Dataset (Dimensions/Variables)
            #    We need to check if 'active_track_id' is a dimensioned array or a scalar
            track_dims = ds['active_track_id'].dims
            
            if len(track_dims) == 0:
                # SCALAR CASE: File has exactly 1 track, and it IS valid (checked above).
                # No slicing needed because the dimension doesn't exist.
                ds_filtered = ds.copy()
            else:
                # ARRAY CASE: File has multiple tracks (or 1 track with explicit dim).
                # We dynamically find the dimension name (usually 'active_track' or 'tracks')
                dim_name = track_dims[0]
                
                # Get integer indices of the valid tracks
                # np.isin returns boolean mask, np.where converts to indices
                valid_indices = np.where(np.isin(all_file_ids, valid_tracks_in_file))[0]
                
                # Slice the dataset along that dimension
                ds_filtered = ds.isel({dim_name: valid_indices})
            
            # 5. Update the Spatial Mask (final_labeled_regions)
            #    Set pixels of REJECTED tracks to 0.
            if 'final_labeled_regions' in ds_filtered:
                mask_da = ds_filtered['final_labeled_regions']
                mask_vals = mask_da.values # Numpy array
                
                # Create a mask where:
                # 1. Pixel value is in valid_tracks_in_file -> Keep it
                # 2. Pixel value is NOT in valid_tracks_in_file -> Set to 0
                new_mask = np.where(np.isin(mask_vals, valid_tracks_in_file), mask_vals, 0)
                
                # Assign back to the dataset safely
                # (We use .values[:] to update the underlying numpy array in-place)
                ds_filtered['final_labeled_regions'].values[:] = new_mask
            
            # --- 6. SAVE WITH MIRRORED STRUCTURE ---
            
            # Calculate the structure we want to preserve (e.g. "1998/05/file.nc")
            # We subtract 'input_root_dir' from the full 'f_path'
            relative_structure = os.path.relpath(f_path, input_root_dir)
            
            # Construct the new full path
            out_path = os.path.join(output_dir, relative_structure)
            
            # Create the necessary sub-folders (e.g. .../tracking/1998/05)
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            
            ds_filtered.to_netcdf(out_path)


def run_postprocessing_year(year: int, raw_tracking_output_dir: str, tracking_output_dir: str, precip_data_var: str, lifted_index_data_var: str, 
                            lat_name: str, lon_name: str, config: object):
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
        config (object): Global configuration object.
    """
    logger.info(f"--- Starting Post-Processing for Year: {year} ---")
    year_data_dir = os.path.join(raw_tracking_output_dir, str(year))
    
    if not os.path.exists(year_data_dir):
        raise FileNotFoundError(f"Raw tracking directory for year {year} not found: {year_data_dir}")

    # Recursive search to find files in subdirectories (e.g., 2020/08/*.nc)
    raw_files = sorted(glob.glob(os.path.join(year_data_dir, "**", "*.nc"), recursive=True))
    
    if not raw_files:
        raise FileNotFoundError(f"No .nc files found in {year_data_dir}")
    
    logger.info(f"Found {len(raw_files)} raw tracking files in {year_data_dir}")

    
    # Ensure yearly output directory exists for NetCDFs
    os.makedirs(tracking_output_dir, exist_ok=True)

    # 2. Locate Environmental Data (LI and Precip)
    li_dir = config.lifted_index_data_directory
    precip_dir = config.precip_data_directory

    all_li = sorted(glob.glob(os.path.join(li_dir, "**", "*.nc"), recursive=True))
    all_precip = sorted(glob.glob(os.path.join(precip_dir, "**", "*.nc"), recursive=True))
    
    # Filter files relevant for this year to optimize search
    li_files_year = [f for f in all_li if str(year) in os.path.basename(f)]
    precip_files_year = [f for f in all_precip if str(year) in os.path.basename(f)]

   # --- STEP 1: Extract Timestep Properties ---
    logger.info("STEP 1: Extracting timestep properties...")
    all_timestep_rows = []
    
    if config.use_multiprocessing and config.number_of_cores > 1:
        logger.info(f"Running extraction in PARALLEL mode ({config.number_of_cores} cores)...")
        with concurrent.futures.ProcessPoolExecutor(max_workers=config.number_of_cores) as executor:
            futures = {
                executor.submit(
                    process_single_timestep, 
                    f, 
                    li_files_year, 
                    precip_files_year, 
                    config,
                    precip_data_var,       # Passed from run_postprocessing_year args
                    lifted_index_data_var, # Passed from run_postprocessing_year args
                    lat_name,              # Passed from run_postprocessing_year args
                    lon_name               # Passed from run_postprocessing_year args
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
                li_files_year, 
                precip_files_year, 
                config,
                precip_data_var,
                lifted_index_data_var,
                lat_name,
                lon_name
            )
            if res:
                all_timestep_rows.extend(res)

    df_timesteps = pd.DataFrame(all_timestep_rows)
    df_timesteps['datetime'] = pd.to_datetime(df_timesteps['datetime'])
    df_timesteps = df_timesteps.sort_values(by=['datetime', 'track_number'])
    
    # Update GLOBAL Timestep CSV
    csv_timestep_path = os.path.join(tracking_output_dir, 'mcs_timestep_properties.csv')
    update_global_csv(df_timesteps, csv_timestep_path, 'datetime', year)

    # --- STEP 2: Aggregate & Analyze ---
    logger.info("STEP 2: Analyzing track properties...")
    
    aggregations = {
        'datetime': ['min', 'max', 'count'],
        'area_km2': ['mean', 'max'],
        'mean_li': ['mean'],
        'mean_precip': ['mean'],
        'max_precip': ['max'],
        'convective_area_km2': ['mean'],
        'stratiform_area_km2': ['mean'],
        'precip_skew': ['mean']
    }
    
    # GroupBy & Aggregation
    df_summary = df_timesteps.groupby('track_number').agg(aggregations)
    df_summary.columns = ['_'.join(col).strip() for col in df_summary.columns.values]
    
    rename_map = {
        'datetime_min': 'start_time',
        'datetime_max': 'end_time',
        'datetime_count': 'duration_steps',
        'area_km2_mean': 'lifetime_mean_area_km2',
        'area_km2_max': 'max_area_km2',
        'mean_li_mean': 'lifetime_mean_LI',
        'mean_precip_mean': 'lifetime_mean_precip',
        'max_precip_max': 'peak_max_precip'
    }
    df_summary = df_summary.rename(columns=rename_map)
    
    # Calculate Kinematics & Volatility
    kinematics = df_timesteps.groupby('track_number').apply(calculate_kinematics)
    volatility = df_timesteps.groupby('track_number').apply(calculate_area_change)
    
    df_summary = df_summary.merge(kinematics, on='track_number')
    df_summary = df_summary.merge(volatility, on='track_number')

    # Update GLOBAL Summary CSV
    csv_summary_path = os.path.join(tracking_output_dir, 'mcs_track_summary.csv')
    update_global_csv(df_summary, csv_summary_path, 'start_time', year)

    # --- STEP 3: Filter ---
    logger.info("STEP 3: Filtering tracks...")
    
    filters = config.postprocessing_filters
    thresh_li = filters.lifted_index_threshold
    thresh_straight = filters.track_straightness_threshold
    thresh_vol = filters.max_area_volatility
    
    logger.info(f"Filtering Criteria: LI < {thresh_li}, Straightness > {thresh_straight}, Volatility < {thresh_vol}")

    
    # Filter Logic
    accepted_mask = (
        (df_summary['lifetime_mean_LI'] < thresh_li) &
        (df_summary['track_straightness'] > thresh_straight) &
        (df_summary['max_area_volatility'] < thresh_vol)
    )
    
    df_accepted = df_summary[accepted_mask]
    df_rejected = df_summary[~accepted_mask]
    
    # Update GLOBAL Accepted/Rejected CSVs
    update_global_csv(df_accepted, os.path.join(tracking_output_dir, 'mcs_track_summary_ACCEPTED.csv'), 'start_time', year)
    update_global_csv(df_rejected, os.path.join(tracking_output_dir, 'mcs_track_summary_REJECTED.csv'), 'start_time', year)
    
    valid_ids = set(df_accepted.index.tolist())
    logger.info(f"Accepted: {len(valid_ids)}, Rejected: {len(df_rejected)}")

    # --- STEP 4: Restore ---
    logger.info("STEP 4: Restoring filtered NetCDF files...")
    if valid_ids:
        restore_filtered_files(raw_files, valid_ids, tracking_output_dir, raw_tracking_output_dir)
        logger.info(f"Restoration complete in {tracking_output_dir}")
    else:
        logger.warning("No tracks passed filtering for this year.")