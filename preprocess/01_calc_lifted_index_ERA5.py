import xarray as xr
import numpy as np
import pandas as pd
import argparse
import logging
from pathlib import Path
import warnings
import multiprocessing
from functools import partial

# --- CONFIGURATION ---
# Adjust these paths to match your ERA5 directory structure
PL_DIR = Path("/reloclim/dkn/data/ERA5/pressure_level") 
SURF_DIR = Path("/reloclim/dkn/data/ERA5/surface")
OUTPUT_DIR = Path("/reloclim/dkn/data/ERA5/lifted_index_final")

# Processing Settings
WARM_SEASON_MONTHS = [5, 6, 7, 8, 9]
SOURCE_LEVELS = [925, 850, 700]
TARGET_LEVEL = 500

# Strict Masking (Matches Pucik 2017 & CORDEX script)
# "We ensured that the selected layer was always above the local topography..."
# Reference: Púčik et al. (2017), Section 2.
TOLERANCE_HPA = 0.0 

# --- 1. VECTORIZED PHYSICS (Bolton 1980) ---

def calculate_dewpoint(e_hpa):
    """
    Calculates Dewpoint (K) from Vapor Pressure (hPa).
    
    Formula derived from the inverse of the Magnus formula for saturation vapor pressure.
    
    Reference:
        Bolton, D. (1980). The computation of equivalent potential temperature. 
        Monthly Weather Review, 108(7), 1046-1053. (Eq. 10 integration)
    """
    e_safe = np.maximum(e_hpa, 0.001)
    ln_e = np.log(e_safe / 6.112)
    numerator = 243.5 * ln_e
    denominator = 17.67 - ln_e
    td_c = numerator / denominator
    return td_c + 273.15

def calculate_tlcl(T, Td):
    """
    Calculates the Temperature at the Lifting Condensation Level (LCL) in Kelvin.
    
    Reference:
        Bolton, D. (1980). The computation of equivalent potential temperature. 
        Monthly Weather Review, 108(7), 1046-1053. (Eq. 15)
    """
    Td_safe = np.maximum(Td, 57.0) 
    denom = (1.0 / (Td_safe - 56.0)) + (np.log(T / Td_safe) / 800.0)
    tlcl = (1.0 / denom) + 56.0
    return tlcl

def calculate_theta_e(T, p_pa, q):
    """
    Calculates Equivalent Potential Temperature (Theta-E) in Kelvin.
    
    Reference:
        Bolton, D. (1980). The computation of equivalent potential temperature. 
        Monthly Weather Review, 108(7), 1046-1053. (Eq. 43)
        
    Notes:
        - Uses T_LCL from Eq. 15.
        - Includes the 0.2854 Poisson constant for dry air.
    """
    # 1. Vapor Pressure
    e_pa = p_pa * q / (0.622 + 0.378 * q)
    e_hpa = e_pa / 100.0
    
    # 2. Dewpoint
    Td = calculate_dewpoint(e_hpa)
    
    # 3. LCL Temperature
    tlcl = calculate_tlcl(T, Td)
    
    # 4. Potential Temperature
    theta = T * (100000.0 / p_pa) ** 0.2854
    
    # 5. Mixing Ratio
    r = q / (1.0 - q)
    
    # 6. Theta-E Calculation (Bolton Eq. 43)
    # Clip exponent argument for numerical stability in extreme/invalid inputs
    exp_arg = (3376.0 / tlcl - 2.54) * r * (1.0 + 0.81 * r)
    exp_arg = np.clip(exp_arg, -30.0, 30.0)
    theta_e = theta * np.exp(exp_arg)
    
    return theta_e

def solve_t500_exact(theta_e_target):
    """
    Iterative solver for Parcel Temperature at 500 hPa along a moist adiabat.
    
    Methodology:
        Inverts the Bolton (1980) Theta-E formula using a Newton-Raphson-like 
        iterative approach to find T such that Theta-E(T, 500hPa, Saturation) 
        equals the source parcel's Theta-E.
        
    Reference for LI Definition:
        Galway, J. G. (1956). The lifted index as a predictor of latent instability. 
        Bulletin of the American Meteorological Society, 37, 528-529.
    """
    p_target = 50000.0 # 500 hPa in Pa
    
    # Create valid mask
    valid_mask = np.isfinite(theta_e_target) & (theta_e_target > 200.0) & (theta_e_target < 500.0)
    te_safe = np.where(valid_mask, theta_e_target, 330.0)
    
    # Initial Guess
    T = np.full_like(te_safe, 250.0)
    
    # Iterative loop (Vectorized, 8 iterations sufficient for convergence < 0.01K)
    for _ in range(8): 
        T = np.clip(T, 150.0, 350.0)
        
        # Saturated Thermodynamics at 500 hPa
        T_c = T - 273.15
        es = 611.2 * np.exp(17.67 * T_c / (T_c + 243.5))
        qs = 0.622 * es / (p_target - 0.378 * es)
        r = qs / (1.0 - qs)
        theta = T * (100000.0 / p_target) ** 0.2854
        
        # Theta-E (Saturated)
        exp_arg = (3376.0 / T - 2.54) * r * (1.0 + 0.81 * r)
        exp_arg = np.clip(exp_arg, -30.0, 30.0)
        
        te_calc = theta * np.exp(exp_arg)
        f_val = te_calc - te_safe
        
        # Derivative (Finite Difference)
        dt = 0.1
        T_next = T + dt
        T_c_n = T_next - 273.15
        es_n = 611.2 * np.exp(17.67 * T_c_n / (T_c_n + 243.5))
        qs_n = 0.622 * es_n / (p_target - 0.378 * es_n)
        r_n = qs_n / (1.0 - qs_n)
        th_n = T_next * (100000.0 / p_target) ** 0.2854
        te_n_calc = th_n * np.exp(np.clip((3376.0/T_next - 2.54)*r_n*(1+0.81*r_n), -30, 30))
        
        df_dt = (te_n_calc - te_calc) / dt
        
        # Update
        step = f_val / (df_dt + 1e-6)
        step = np.clip(step, -10.0, 10.0)
        T = T - step
            
    return np.where(valid_mask, T, np.nan)

# --- 2. WORKER LOGIC (MONTH-BASED) ---

def process_month(task_info, output_base_dir):
    """
    Loads one month of ERA5 data, Calculates LI, Saves hourly files.
    """
    year, month = task_info
    month_str = f"{year}-{month:02d}"
    
    # Define File Paths
    # Pattern: YYYY-MM_LI.nc and YYYY-MM_SP.nc
    pl_file = PL_DIR / f"{month_str}_LI.nc"
    surf_file = SURF_DIR / f"{month_str}_SP.nc"
    
    if not pl_file.exists() or not surf_file.exists():
        return f"Skipping {month_str} (Files not found)"

    try:
        # 1. Load Data
        ds_pl = xr.open_dataset(pl_file)
        ds_surf = xr.open_dataset(surf_file)
        
        # 2. FLIP LATITUDE 
        # ERA5 is Descending (90 -> -90). We sort to Ascending (-90 -> 90)
        ds_pl = ds_pl.sortby("latitude")
        ds_surf = ds_surf.sortby("latitude")
        
        # 3. Check Alignment
        common_times = np.intersect1d(ds_pl.valid_time, ds_surf.valid_time)
        if len(common_times) == 0:
            return f"Error {month_str}: No common times found."
            
        ds_pl = ds_pl.sel(valid_time=common_times)
        ds_surf = ds_surf.sel(valid_time=common_times)
        
        # 4. Extract Numpy Arrays
        sp_pa = ds_surf["sp"].values 
        
        # Environment (500 hPa)
        ds_env = ds_pl.sel(pressure_level=TARGET_LEVEL)
        t_env_500 = ds_env["t"].values
        q_env_500 = ds_env["q"].values
        
        # 5. Calculate LI (Loop over source levels)
        li_candidates = []
        
        for p_src in SOURCE_LEVELS:
            p_src_pa = float(p_src * 100.0)
            
            ds_src = ds_pl.sel(pressure_level=p_src)
            t_src = ds_src["t"].values
            q_src = ds_src["q"].values
            
            # --- PHYSICS CALCULATION ---
            te_src = calculate_theta_e(t_src, p_src_pa, q_src)
            t_parcel_500 = solve_t500_exact(te_src)
            
            # Parcel Virtual Temp (Saturated)
            T_c_par = t_parcel_500 - 273.15
            es_parcel = 611.2 * np.exp(17.67 * T_c_par / (T_c_par + 243.5))
            qs_parcel = 0.622 * es_parcel / (50000.0 - 0.378 * es_parcel)
            # Doswell and Rasmussen (1994) Virtual Temperature Correction
            tv_parcel = t_parcel_500 * (1.0 + 0.61 * qs_parcel)
            
            # Environment Virtual Temp
            tv_env = t_env_500 * (1.0 + 0.61 * q_env_500)
            
            # LI Calculation
            li_level = tv_env - tv_parcel
            
            # Masking (Strict)
            limit = sp_pa + (TOLERANCE_HPA * 100.0)
            is_underground = p_src_pa > limit
            li_level[is_underground] = np.nan
            
            li_candidates.append(li_level)
            
        # 6. Minimum LI
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            li_final = np.nanmin(np.stack(li_candidates, axis=0), axis=0)
            
        # 7. Save Hourly Files
        coords = {
            "valid_time": common_times,
            "latitude": ds_pl.latitude,   # Sorted Ascending
            "longitude": ds_pl.longitude
        }
        
        out_month_dir = Path(output_base_dir) / f"{year}" / f"{month:02d}"
        out_month_dir.mkdir(parents=True, exist_ok=True)
        
        for i, t_val in enumerate(common_times):
            ts = pd.to_datetime(t_val)
            filename = out_month_dir / f"lifted_index_{ts.strftime('%Y%m%dT%H')}.nc"
            
            # Add time dimension back: (1, Lat, Lon)
            data_slice = li_final[i, :, :][np.newaxis, :, :]
            
            da_out = xr.DataArray(
                data_slice,
                coords={
                    "time": [t_val], 
                    "lat": coords["latitude"],
                    "lon": coords["longitude"]
                },
                dims=("time", "latitude", "longitude"),
                name="LI",
                attrs={
                    "units": "K", 
                    "long_name": "Most Unstable LI (Bolton, Strict)",
                    "description": "Calculated from ERA5. Pucik et al. (2017) methodology."
                }
            )
            
            da_out.to_netcdf(filename)
            
        return f"Done: {month_str}"

    except Exception as e:
        import traceback
        traceback.print_exc()
        return f"Error {month_str}: {e}"

# --- 3. MANAGER ---

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start_year", type=int, default=1998)
    parser.add_argument("--end_year", type=int, default=2025)
    parser.add_argument("--cores", type=int, default=20)
    parser.add_argument("--output_dir", type=str, default="/reloclim/dkn/data/ERA5/lifted_index_final")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
    
    tasks = []
    for year in range(args.start_year, args.end_year + 1):
        for month in WARM_SEASON_MONTHS:
            tasks.append((year, month))
            
    logging.info(f"--- Starting Processing: {len(tasks)} Months ---")
    logging.info(f"    Output: {args.output_dir}")
    
    worker = partial(process_month, output_base_dir=args.output_dir)
    
    if args.debug:
        for t in tasks:
            print(f"Processing {t}...")
            print(worker(t))
    else:
        safe_cores = min(args.cores, len(tasks))
        with multiprocessing.Pool(safe_cores) as pool:
            for res in pool.imap_unordered(worker, tasks):
                logging.info(res)

if __name__ == "__main__":
    main()