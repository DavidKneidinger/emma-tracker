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
VAR_MAPPING = {
    "T": "T",                
    "QV": "QV",              
    "PS": "PS"               
}

LEVELS = [500, 700, 850, 925]
WARM_SEASON_MONTHS = [5, 6, 7, 8, 9]

# Tolerance: 0.0 matches Pucik2017 approach.
TOLERANCE_HPA = 0.0 

# --- 1. VECTORIZED PHYSICS (Bolton 1980) ---
# Optimized for pure Numpy arrays

def calculate_dewpoint(e_hpa):
    """Calculates Dewpoint (K) from Vapor Pressure (hPa)."""
    e_safe = np.maximum(e_hpa, 0.001)
    ln_e = np.log(e_safe / 6.112)
    numerator = 243.5 * ln_e
    denominator = 17.67 - ln_e
    td_c = numerator / denominator
    return td_c + 273.15

def calculate_tlcl(T, Td):
    """Calculates the Temperature at the Lifting Condensation Level (LCL) in Kelvin."""
    Td_safe = np.maximum(Td, 57.0) 
    denom = (1.0 / (Td_safe - 56.0)) + (np.log(T / Td_safe) / 800.0)
    tlcl = (1.0 / denom) + 56.0
    return tlcl

def calculate_theta_e(T, p_pa, q):
    """Calculates Equivalent Potential Temperature (Theta-E) in Kelvin."""
    e_pa = p_pa * q / (0.622 + 0.378 * q)
    e_hpa = e_pa / 100.0
    
    Td = calculate_dewpoint(e_hpa)
    tlcl = calculate_tlcl(T, Td)
    
    theta = T * (100000.0 / p_pa) ** 0.2854
    r = q / (1.0 - q)
    
    exp_arg = (3376.0 / tlcl - 2.54) * r * (1.0 + 0.81 * r)
    exp_arg = np.clip(exp_arg, -30.0, 30.0)
    theta_e = theta * np.exp(exp_arg)
    
    return theta_e

def solve_t500_exact(theta_e_target):
    """Iterative solver for Parcel Temperature at 500 hPa along a moist adiabat."""
    p_target = 50000.0
    
    valid_mask = np.isfinite(theta_e_target) & (theta_e_target > 200.0) & (theta_e_target < 500.0)
    te_safe = np.where(valid_mask, theta_e_target, 330.0)
    
    T = np.full_like(te_safe, 250.0)
    
    for _ in range(8): 
        T = np.clip(T, 150.0, 350.0)
        
        T_c = T - 273.15
        es = 611.2 * np.exp(17.67 * T_c / (T_c + 243.5))
        qs = 0.622 * es / (p_target - 0.378 * es)
        r = qs / (1.0 - qs)
        theta = T * (100000.0 / p_target) ** 0.2854
        
        exp_arg = (3376.0 / T - 2.54) * r * (1.0 + 0.81 * r)
        exp_arg = np.clip(exp_arg, -30.0, 30.0)
        
        te_calc = theta * np.exp(exp_arg)
        f_val = te_calc - te_safe
        
        # Derivative
        dt = 0.1
        T_next = T + dt
        T_c_n = T_next - 273.15
        es_n = 611.2 * np.exp(17.67 * T_c_n / (T_c_n + 243.5))
        qs_n = 0.622 * es_n / (p_target - 0.378 * es_n)
        r_n = qs_n / (1.0 - qs_n)
        th_n = T_next * (100000.0 / p_target) ** 0.2854
        te_n_calc = th_n * np.exp(np.clip((3376.0/T_next - 2.54)*r_n*(1+0.81*r_n), -30, 30))
        
        df_dt = (te_n_calc - te_calc) / dt
        
        step = f_val / (df_dt + 1e-6)
        step = np.clip(step, -10.0, 10.0)
        T = T - step
            
    return np.where(valid_mask, T, np.nan)

# --- 2. WORKER LOGIC ---

def save_hourly_frames(ds_obj, prefix, output_base_dir):
    """
    Saves hourly frames preserving the time dimension and exact timestamp (including minutes).
    Uses a full Dataset object to preserve all auxiliary variables (lat, lon, rotated_pole).
    Output Format: prefix_YYYYMMDDTHHMM.nc
    """
    # Iterate using index to allow slicing that PRESERVES the dimension
    for i in range(len(ds_obj.time)):
        # Extract the exact timestamp from the data (e.g. 1979-01-01 00:30:00)
        ts = pd.to_datetime(ds_obj.time.values[i])
        
        out_dir = Path(output_base_dir) / f"{ts.year}" / f"{ts.month:02d}"
        out_dir.mkdir(parents=True, exist_ok=True)
        
        # New Format: YYYYMMDDTHHMM (includes minutes)
        filename = out_dir / f"{prefix}_{ts.strftime('%Y%m%dT%H%M')}.nc"
        
        if not filename.exists():
            # .isel(time=[i]) selects the index but returns a Dataset 
            # with shape (1, y, x), preserving time dim and all variables
            ds_out = ds_obj.isel(time=[i])
            ds_out.to_netcdf(filename, engine='netcdf4')

def process_single_day(day_info, file_paths, output_dir, cordex_run):
    """
    Worker function: Opens files, computes LI for ONE day, saves result as Dataset.
    """
    day_str, year = day_info
    
    try:
        # 1. Open Reference to get the exact time axis and Grid
        # We use this ds_ref as the template for the output
        ds_ref = xr.open_dataset(file_paths["T925p"]).sel(time=day_str)
        target_time = ds_ref.time
        
        # 2. Load Data for this day
        data = {}
        
        # Surface Pressure
        ds_ps = xr.open_dataset(file_paths["PS"])
        ds_ps = ds_ps.interp(time=target_time, method="linear", kwargs={"fill_value": "extrapolate"})
        data["sp"] = ds_ps[VAR_MAPPING["PS"]].values 

        # Upper Levels
        for level in LEVELS:
            # Temperature
            path_t = file_paths[f"T{level}p"]
            ds_t = xr.open_dataset(path_t).interp(time=target_time, method="linear")
            data[f"t_{level}"] = ds_t[VAR_MAPPING["T"]].values
            
            # Humidity
            path_q = file_paths[f"QV{level}p"]
            ds_q = xr.open_dataset(path_q).interp(time=target_time, method="linear")
            data[f"q_{level}"] = ds_q[VAR_MAPPING["QV"]].values

        # 3. Compute LI
        t_env_500 = data["t_500"]
        q_env_500 = data["q_500"]
        sp_pa = data["sp"]
        
        li_candidates = []
        
        for p_src in [925, 850, 700]:
            p_src_pa = float(p_src * 100.0)
            t_src = data[f"t_{p_src}"]
            q_src = data[f"q_{p_src}"]
            
            # --- PHYSICS ---
            te_src = calculate_theta_e(t_src, p_src_pa, q_src)
            t_parcel_500 = solve_t500_exact(te_src)
            
            # Parcel Virtual Temp (Saturated)
            T_c_par = t_parcel_500 - 273.15
            es_parcel = 611.2 * np.exp(17.67 * T_c_par / (T_c_par + 243.5))
            qs_parcel = 0.622 * es_parcel / (50000.0 - 0.378 * es_parcel)
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
            
        # 4. Minimum LI
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # This preserves the time dimension (axis 0 of inputs matches axis 0 of output)
            li_final = np.nanmin(np.stack(li_candidates, axis=0), axis=0)
            
        # 5. Create Final Dataset
        # We start with the reference dataset structure to keep rlat, rlon, lat, lon, rotated_pole
        ds_out = ds_ref.copy(deep=False)
        
        # Remove the original data variable (e.g. Temperature) but keep coordinates
        # We assume the variable name matches VAR_MAPPING["T"] based on ds_ref source
        ds_out = ds_out.drop_vars(list(ds_out.data_vars))

        # Identify spatial dimensions
        spatial_dims = [d for d in ds_ref.dims if 'time' not in d]

        # Add LI Variable
        ds_out["LI"] = (("time", spatial_dims[0], spatial_dims[1]), li_final)
        
        # Set Attributes
        ds_out["LI"].attrs = {
            "units": "K", 
            "long_name": "Most Unstable LI (Bolton, Strict)",
            "description": "Calculated from ERA5. Pucik et al. (2017) methodology."
        }
        
        # Ensure grid mapping attribute is present if rotated_pole exists
        if "rotated_pole" in ds_out.variables:
            ds_out["LI"].attrs["grid_mapping"] = "rotated_pole"

        # Save
        output_str = f"lifted_index_{cordex_run}"
        save_hourly_frames(ds_out, output_str, output_dir)
        
        return f"Done: {day_str}"

    except Exception as e:
        import traceback
        traceback.print_exc()
        return f"Error {day_str}: {e}"


# --- 3. MAIN PROCESS ---

def find_file_for_year(base_dir, var_prefix, year):
    search_dir = Path(base_dir) / var_prefix
    if not search_dir.exists():
        raise FileNotFoundError(f"Directory not found: {search_dir}")
    
    pattern = f"*_{year}010100-*.ncz"
    found_files = list(search_dir.glob(pattern))
    if not found_files:
        pattern_nc = f"*_{year}010100-*.nc"
        found_files = list(search_dir.glob(pattern_nc))

    if not found_files:
        raise FileNotFoundError(f"No file found for {var_prefix}/{year}")
    return found_files[0]

def process_year(year, args):
    logging.info(f"--- Setting up Year {year} ---")
    
    # 1. Locate Files
    try:
        paths = {}
        paths["PS"] = find_file_for_year(args.input_dir, "PS", year)
        # Ref for time
        paths["T925p"] = find_file_for_year(args.input_dir, "T925p", year)
        paths["QV925p"] = find_file_for_year(args.input_dir, "QV925p", year)
        
        for level in [500, 700, 850]:
            paths[f"T{level}p"] = find_file_for_year(args.input_dir, f"T{level}p", year)
            paths[f"QV{level}p"] = find_file_for_year(args.input_dir, f"QV{level}p", year)
    except FileNotFoundError as e:
        logging.error(e)
        return

    # 2. Find Warm Season Days
    with xr.open_dataset(paths["T925p"]) as ds:
        all_days = np.unique(ds.time.dt.floor("D"))
        is_warm = np.isin(pd.to_datetime(all_days).month, WARM_SEASON_MONTHS)
        warm_days = all_days[is_warm]
        
    warm_day_strs = [str(d).split('T')[0] for d in warm_days]
    logging.info(f"Queuing {len(warm_day_strs)} days...")
    
    # 3. Create Tasks
    output_dir = args.output_basedir + f"{args.cordex_run}"

    worker = partial(process_single_day, file_paths=paths, output_dir=output_dir, cordex_run=args.cordex_run)
    tasks = [(d, year) for d in warm_day_strs]
    
    # 4. Execute
    if args.debug:
        logging.info("Running in SERIAL DEBUG MODE")
        for t in tasks:
            print(f"Processing {t[0]}...")
            print(worker(t))
    else:
        logging.info(f"Running in PARALLEL MODE ({args.cores} cores)")
        with multiprocessing.Pool(args.cores) as pool:
            for i, res in enumerate(pool.imap_unordered(worker, tasks)):
                if i % 10 == 0:
                    logging.info(f"Progress: {i}/{len(tasks)} days")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, default="/reloclim/het/ICHISTL02/")
    parser.add_argument("--output_basedir", type=str, default="/reloclim/dkn/euro-cordex/data/lifted_index/")
    parser.add_argument("--cordex_run", type=str, default="ICHISTL02")
    parser.add_argument("--start_year", type=int, default=1950)
    parser.add_argument("--end_year", type=int, default=1997)
    parser.add_argument("--cores", type=int, default=10)
    parser.add_argument("--debug", action="store_true", help="Enable serial debug mode")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
    
    for year in range(args.start_year, args.end_year + 1):
        process_year(year, args)

if __name__ == "__main__":
    main()