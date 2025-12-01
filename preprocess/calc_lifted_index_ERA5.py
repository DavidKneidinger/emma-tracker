import xarray as xr
import numpy as np
import pandas as pd
import os
import argparse
import logging
import xesmf as xe
import multiprocessing
from collections import defaultdict
from pathlib import Path

# --- Constants ---
RD = 287.05  # Gas constant for dry air, J/(kg*K)
G = 9.81  # Gravity, m/s^2
GAMMA_M = (
    6.0  # Assumed moist adiabatic lapse rate in K/km, following Pucik et al. (2017)
)


def compute_li_for_parcel(T_src, q_src, T_env500, sp_hpa, p_src_hpa):
    """
    Computes the Lifted Index (LI) for a single air parcel in a vectorized manner.

    This calculation is based on the methodology described in Pucik et al. (2017),
    "Future Changes in European Severe Convection Environments in a Regional
    Climate Model Ensemble."

    Args:
        T_src (np.ndarray): Temperature at the source level (K).
        q_src (np.ndarray): Specific humidity at the source level (kg/kg).
        T_env500 (np.ndarray): Environmental temperature at 500 hPa (K).
        sp_hpa (np.ndarray): Surface pressure (hPa).
        p_src_hpa (float): Pressure of the source level (hPa).

    Returns:
        np.ndarray: The calculated Lifted Index (K). Returns np.nan for grid points
                    where the source level is at or below the surface pressure.
    """
    # Calculate virtual temperature of the source parcel for all points
    t_virtual_src = T_src * (1 + 0.61 * q_src)

    # Estimate the height difference to 500 hPa using the hypsometric equation
    delta_z_m = (RD * t_virtual_src / G) * np.log(p_src_hpa / 500.0)

    # Estimate the parcel's temperature at 500 hPa by lifting it along a moist adiabat
    t_parcel_500 = t_virtual_src - (GAMMA_M * (delta_z_m / 1000.0))

    # LI is the difference between the environment and the lifted parcel temperature
    li = T_env500 - t_parcel_500

    # Use np.where to conditionally mask out values where the parcel is underground.
    return np.where(sp_hpa <= p_src_hpa, np.nan, li)


def process_month(month, file_paths, ds_source_grid, ds_target_grid, weights_path, args, lat_min, lat_max, lon_min, lon_max):
    """
    Worker function to process all data for a single month.
    This function is designed to be called by a multiprocessing pool.
    """
    pid = os.getpid()
    
    # --- Initialize the regridder within the worker process ---
    regridder = xe.Regridder(
        ds_source_grid,
        ds_target_grid,
        "bilinear",
        reuse_weights=True,  # The weights file is guaranteed to exist
        filename=weights_path,
    )
    logging.info(f"[PID {pid}] Regridder initialized for month {month}.")
    
    logging.info(f"[PID {pid}] Processing data for {month}...")

    # --- EFFICIENT CHECK: Determine if all output files for this month already exist ---
    year, month_num = map(int, month.split("-"))
    try:
        days_in_month = pd.Period(month).days_in_month
        expected_hours = pd.date_range(
            start=f"{month}-01", periods=days_in_month * 24, freq="h"
        )
    except ValueError:
        logging.error(f"[PID {pid}] Invalid month string format: {month}. Skipping.")
        return

    output_files_exist = []
    for timestamp in expected_hours:
        output_path = Path(args.output_dir) / timestamp.strftime("%Y/%m")
        output_filename = (
            output_path / f"lifting_index_{timestamp.strftime('%Y%m%dT%H')}.nc"
        )
        output_files_exist.append(output_filename.exists())

    if all(output_files_exist):
        logging.info(f"[PID {pid}] All output files for {month} already exist. Skipping.")
        return

    try:
        with xr.open_dataset(
            file_paths["pl"], chunks={"valid_time": 24}
        ) as ds_pl, xr.open_dataset(
            file_paths["sp"], chunks={"valid_time": 24}
        ) as ds_sp:

            ds_pl = ds_pl.rename({"valid_time": "time"})
            ds_sp = ds_sp.rename({"valid_time": "time"})

            ds_pl_aligned, ds_sp_aligned = xr.align(ds_pl, ds_sp, join="inner")

            if "expver" in ds_pl_aligned:
                ds_pl_aligned = ds_pl_aligned.drop_vars("expver")

            if ds_pl_aligned.time.size == 0:
                logging.warning(f"[PID {pid}] No matching time steps found for {month}. Skipping.")
                return
            logging.info(
                f"[PID {pid}] Aligned datasets for {month}, found {ds_pl_aligned.time.size} matching time steps."
            )

            ds_pl_cropped = ds_pl_aligned.sortby("latitude").sel(
                latitude=slice(lat_min, lat_max), longitude=slice(lon_min, lon_max)
            )
            ds_sp_cropped = ds_sp_aligned.sortby("latitude").sel(
                latitude=slice(lat_min, lat_max), longitude=slice(lon_min, lon_max)
            )

            sp_hpa = ds_sp_cropped.sp / 100.0
            T_env500 = ds_pl_cropped.t.sel(pressure_level=500)

            li_candidates = []
            for p_src in [925, 850, 700]:
                T_src = ds_pl_cropped.t.sel(pressure_level=p_src)
                q_src = ds_pl_cropped.q.sel(pressure_level=p_src)
                li_level = xr.apply_ufunc(
                    compute_li_for_parcel,
                    T_src,
                    q_src,
                    T_env500,
                    sp_hpa,
                    p_src,
                    dask="parallelized",
                    output_dtypes=[float],
                )
                li_candidates.append(li_level)

            li_all_levels = xr.concat(
                li_candidates, dim=pd.Index([925, 850, 700], name="pressure_level")
            )
            li_final_native = li_all_levels.min(dim="pressure_level", skipna=True)

            logging.info(f"[PID {pid}] Remapping LI field for {month}...")

            li_to_regrid = li_final_native.rename(
                {"latitude": "lat", "longitude": "lon"}
            )
            li_to_regrid = li_to_regrid.transpose("time", "lat", "lon")
            li_regridded = regridder(li_to_regrid)

            # --- Save results to hourly files ---
            for time_step in li_regridded.time.values:
                timestamp = pd.to_datetime(time_step)
                output_path = Path(args.output_dir) / timestamp.strftime("%Y/%m")
                output_filename = (
                    output_path / f"lifting_index_{timestamp.strftime('%Y%m%dT%H')}.nc"
                )

                if output_filename.exists():
                    continue

                output_path.mkdir(parents=True, exist_ok=True)

                ds_hour = li_regridded.sel(time=slice(timestamp, timestamp))
                ds_out = ds_hour.to_dataset(name="LI")
                
                # Clean up dataset before saving
                vars_to_drop = ['number', 'expver']
                for var in vars_to_drop:
                    if var in ds_out:
                        ds_out = ds_out.drop_vars(var)
                
                ds_out['LI'].attrs['units'] = 'K'
                ds_out.attrs = {
                    "Title": "Remapped Lifted Index calculated from ERA5",
                    "Description": "Most unstable LI from 925, 850, and 700 hPa source parcels.",
                    "Source": "ERA5",
                    "Method": "Pucik et al. (2017), J. Climate",
                    "units": "K",
                    "Author": "David Kneidinger",
                    "Email": "<david.kneidinger@uni-graz.at>",
                }
                ds_out.to_netcdf(output_filename)
                logging.info(f"[PID {pid}] Saved: {output_filename}")

    except Exception as e:
        logging.error(f"[PID {pid}] An error occurred while processing {month}: {e}")


def parse_arguments():
    """Parses command-line arguments for the LI calculation script."""
    parser = argparse.ArgumentParser(
        description="Calculate, remap, and save Lifted Index (LI) from ERA5 data in parallel."
    )
    parser.add_argument(
        "--pl_dir",
        type=str,
        default="/reloclim/dkn/data/ERA5/pressure_level/",
        help="Directory containing monthly ERA5 pressure level ('YYYY-MM_LI.nc') files.",
    )
    parser.add_argument(
        "--sp_dir",
        type=str,
        default="/reloclim/dkn/data/ERA5/surface/",
        help="Directory containing monthly ERA5 surface pressure ('YYYY-MM_SP.nc') files.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/reloclim/dkn/data/ERA5/lifting_index",
        help="Base directory where the output files will be saved in a YYYY/MM structure.",
    )
    parser.add_argument(
        "--target_grid_file",
        type=str,
        default="/reloclim/dkn/data/IMERG/1998/01/3B-HHR.MS.MRG.3IMERG.V07B_19980101_0000.nc",
        help="Path to a sample NetCDF file that defines the target grid for remapping.",
    )
    parser.add_argument(
        "--weights_dir",
        type=str,
        default="./remapping_weights",
        help="Directory to store (or load from) the xesmf remapping weights file.",
    )
    parser.add_argument(
        "--start_year", type=int, default=2000, help="The first year to process."
    )
    parser.add_argument(
        "--end_year", type=int, default=2024, help="The last year to process."
    )
    parser.add_argument(
        "--num_cores",
        type=int,
        default=1,
        help="Number of CPU cores to use for parallel processing.",
    )
    return parser.parse_args()


def main():
    """
    Main script to process ERA5 data, calculate Lifted Index (LI), remap it
    to a target grid, and save the results into hourly, structured NetCDF files.
    """
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    args = parse_arguments()

    # --- 1. Setup Target Domain ---
    logging.info("--- Setting up target domain ---")
    try:
        ds_target_grid = xr.open_dataset(args.target_grid_file)
    except FileNotFoundError:
        logging.error(f"Target grid file not found at: {args.target_grid_file}")
        return

    lat_min, lat_max = ds_target_grid.lat.min().item(), ds_target_grid.lat.max().item()
    lon_min, lon_max = ds_target_grid.lon.min().item(), ds_target_grid.lon.max().item()
    logging.info(
        f"Target domain set to LON: [{lon_min:.2f}, {lon_max:.2f}], LAT: [{lat_min:.2f}, {lat_max:.2f}]"
    )

    # --- 2. Find and group all input files by month ---
    files_by_month = defaultdict(dict)
    years_to_process = range(args.start_year, args.end_year + 1)

    for year in years_to_process:
        for month_num in range(1, 13):
            month_str = f"{year}-{month_num:02d}"
            pl_file = os.path.join(args.pl_dir, f"{month_str}_LI.nc")
            sp_file = os.path.join(args.sp_dir, f"{month_str}_SP.nc")
            if os.path.exists(pl_file) and os.path.exists(sp_file):
                files_by_month[month_str]["pl"] = pl_file
                files_by_month[month_str]["sp"] = sp_file

    if not files_by_month:
        logging.error("No matching files found for the given year range.")
        return
    logging.info(f"Found {len(files_by_month)} months to process.")

    # --- 3. Prepare for Regridding (main process) ---
    logging.info("--- Preparing for regridding ---")
    first_month = next(iter(files_by_month))
    first_month_paths = files_by_month[first_month]
    
    with xr.open_dataset(first_month_paths["pl"]) as ds_pl_sample, xr.open_dataset(
        first_month_paths["sp"]
    ) as ds_sp_sample:
        ds_pl_cropped_s = ds_pl_sample.sortby("latitude").sel(
            latitude=slice(lat_min, lat_max), longitude=slice(lon_min, lon_max)
        )
    ds_source_grid = xr.Dataset(
        {"lat": ds_pl_cropped_s.latitude, "lon": ds_pl_cropped_s.longitude}
    )

    Path(args.weights_dir).mkdir(parents=True, exist_ok=True)
    weights_path = os.path.join(args.weights_dir, "bilinear_era5_to_target.nc")
    
    # Create the weights file if it doesn't exist. This is done once in the main process
    # to avoid race conditions where multiple workers try to create it at the same time.
    logging.info("Ensuring remapping weights file exists...")
    _ = xe.Regridder(
        ds_source_grid,
        ds_target_grid,
        "bilinear",
        reuse_weights=os.path.exists(weights_path),
        filename=weights_path,
    )
    logging.info(f"Regridder weights are available at: {weights_path}")

    # --- 4. Process each month in parallel ---
    logging.info(f"--- Starting parallel processing on {args.num_cores} cores ---")
    tasks = [
        (month, paths, ds_source_grid, ds_target_grid, weights_path, args, lat_min, lat_max, lon_min, lon_max)
        for month, paths in files_by_month.items()
    ]

    with multiprocessing.Pool(processes=args.num_cores) as pool:
        pool.starmap(process_month, tasks)

    logging.info("--- All tasks dispatched. Preprocessing complete. ---")


if __name__ == "__main__":
    main()

