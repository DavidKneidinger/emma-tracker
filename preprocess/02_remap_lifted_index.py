import xarray as xr
import numpy as np
import warnings
from pathlib import Path
import multiprocessing
import xesmf as xe
import sys
import os
import pandas as pd

# --- 1. CONFIGURATION ---

# DEBUG MODE: Set True to run on 1 core and see full error traces
SERIAL_MODE = False
NUM_CORES = 20

# Directories
INPUT_BASE = Path("/reloclim/dkn/data/ERA5/lifted_index_final")
OUTPUT_BASE = Path("/reloclim/dkn/data/ERA5/lifted_index_corr_remap")

# Weights File (The one you verified with ncdump)
WEIGHTS_FILE = "./remapping_weights/bilinear_era5_to_target.nc"

# Processing Scope
YEARS = list(np.arange(1998, 2025))
MONTHS = ['05', '06', '07', '08', '09']

# Target Grid Definition (IMERG 0.1 deg)
# This is required to define the shape of the output file
TARGET_LAT_MIN, TARGET_LAT_MAX = 29.95, 69.95
TARGET_LON_MIN, TARGET_LON_MAX = -20.05, 40.05
GRID_STEP = 0.1

# --- 2. UTILITIES ---

def get_target_grid():
    """
    Defines the target IMERG grid coordinates.
    This tells xesmf the shape and coordinates of the output.
    """
    # Use float64 to ensure precision matches the weights file generation
    lats = np.arange(TARGET_LAT_MIN, TARGET_LAT_MAX + 0.0001, GRID_STEP)
    lons = np.arange(TARGET_LON_MIN, TARGET_LON_MAX + 0.0001, GRID_STEP)
    
    ds_target = xr.Dataset(
        coords={
            "lat": (["lat"], lats),
            "lon": (["lon"], lons),
        }
    )
    return ds_target

# --- 3. WORKER FUNCTION ---

def process_month_remapping(task_tuple):
    year, month = task_tuple
    
    in_dir = INPUT_BASE / str(year) / month
    out_dir = OUTPUT_BASE / str(year) / month
    
    # 1. Validation
    if not in_dir.exists():
        if SERIAL_MODE: print(f"[{year}-{month}] Input dir missing: {in_dir}")
        return
    
    files = sorted(list(in_dir.glob("*.nc")))
    if not files:
        if SERIAL_MODE: print(f"[{year}-{month}] No .nc files found.")
        return

    # Create output directory
    out_dir.mkdir(parents=True, exist_ok=True)

    if SERIAL_MODE: print(f"[{year}-{month}] Processing {len(files)} files...")

    # --- 2. SETUP REGRIDDER (Once per month/worker) ---
    try:
        # Load one file to define the SOURCE grid structure
        ds_sample = xr.open_dataset(files[0])
        
        # Standardize source names for xesmf (it requires 'lat', 'lon')
        if 'latitude' in ds_sample.coords:
            ds_sample = ds_sample.rename({'latitude': 'lat', 'longitude': 'lon'})
            
        ds_target = get_target_grid()
        
        # Initialize Regridder
        # CRITICAL: reuse_weights=True loads your existing .nc file
        regridder = xe.Regridder(
            ds_sample, 
            ds_target, 
            "bilinear", 
            filename=WEIGHTS_FILE, 
            reuse_weights=True
        )
    except Exception as e:
        print(f"[{year}-{month}] Regridder Init Failed: {e}")
        if SERIAL_MODE: raise e
        return

    # --- 3. FILE LOOP ---
    count = 0
    for f in files:
        out_file = out_dir / f.name
        
        # Skip if done
        if out_file.exists():
            continue
            
      
        with xr.open_dataset(f) as ds:
            # A. Rename Input Coordinates to match Regridder expectation
            rename_in = {}
            if 'latitude' in ds.coords: rename_in['latitude'] = 'lat'
            if 'longitude' in ds.coords: rename_in['longitude'] = 'lon'
            if rename_in:
                ds = ds.rename(rename_in)

            # B. Regrid
            # We assume the variable is 'LI' or take the first variable
            var_name = list(ds.data_vars)[0]
            da_native = ds[var_name]
            
            # Apply weights
            da_remap = regridder(da_native)

            # C. Fix Coordinates & Dimensions (Enforce time, lat, lon)
            
            # 1. Handle Time Dimension
            # If time was lost (because input was 2D), restore it from coordinates
            if 'time' not in da_remap.dims:
                time_val = None
                if 'valid_time' in ds.coords:
                    time_val = ds['valid_time'].values
                elif 'time' in ds.coords:
                    time_val = ds['time'].values
                
                if time_val is not None:
                    # Ensure it's a list/array for expansion
                    if np.ndim(time_val) == 0: time_val = [time_val]
                    da_remap = da_remap.expand_dims(time=time_val)

            # 2. Convert to Dataset
            ds_out = da_remap.to_dataset(name='LI')
            
            # 3. Final Renaming (Strict enforcement)
            final_rename = {}
            
            # Check lat/lon
            if 'latitude' in ds_out.coords: final_rename['latitude'] = 'lat'
            if 'longitude' in ds_out.coords: final_rename['longitude'] = 'lon'
            
            # Check time
            if 'valid_time' in ds_out.coords: final_rename['valid_time'] = 'time'
            
            # Sometimes xesmf/xarray creates a 'dim_0' if names conflict
            for dim in ds_out.dims:
                if dim not in ['lat', 'lon', 'time', 'bnds']:
                    if ds_out.dims[dim] == 1: # Likely the time dim
                            final_rename[dim] = 'time'

            if final_rename:
                ds_out = ds_out.drop(['valid_time'])
            
            # 4. Transpose to standard order (time, lat, lon)
            if 'time' in ds_out.dims:
                ds_out = ds_out.transpose('time', 'lat', 'lon')

            # D. Add Attributes
            ds_out['LI'].attrs = {
                'units': 'K',
                'long_name': 'Lifted Index',
                'description': 'Pucik et al. (2017). Remapped to IMERG (0.1 deg).'
            }
            
            # E. Save
            ds_out.to_netcdf(out_file)
            count += 1

    print(f"[{year}-{month}] Finished. Remapped {count} files.")

# --- 4. MAIN ---

def main():
    tasks = []
    for year in YEARS:
        for month in MONTHS:
            tasks.append((year, month))
    
    if SERIAL_MODE:
        print(f"--- DEBUG MODE: Processing {len(tasks)} months sequentially ---")
        for task in tasks:
            process_month_remapping(task)
    else:
        print(f"--- PARALLEL MODE: Processing {len(tasks)} months on {NUM_CORES} cores ---")
        with multiprocessing.Pool(processes=NUM_CORES) as pool:
            pool.map(process_month_remapping, tasks)

    print("--- Remapping Complete ---")

if __name__ == "__main__":
    main()