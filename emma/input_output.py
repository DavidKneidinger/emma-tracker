"""
emma/input_output.py

Input/Output, Data Loading, Unit Conversion, and CF-Compliant NetCDF Export Module.

This module provides the central I/O pipeline for the EMMA tracking framework, including:
1. Smart directory scanning and task list construction across dataset time slices.
2. Robust dataset loading with strict 1D/2D grid coordinate validation.
3. Unit conversion utilities for primary and environmental variables.
4. Standardized dataset encoding and compression logic optimized for tools like ncview.
5. CF-compliant output export for detection, tracking, and post-processing NetCDF files.
"""

import xarray as xr
import numpy as np
import pandas as pd
import os
import glob
import datetime
import json
import re
import sys
import logging
from emma.grid_manager import build_grid_info

logger = logging.getLogger(__name__)


def build_task_list(
    main_var_dir,
    main_var_template,
    env_var_dir=None,
    env_var_template=None,
    years=None,
    months=None,
    dt_hours=1.0,
):
    """
    Scans data directories using filename templates to build a list of processing tasks.

    This function utilizes a "Smart Pre-Filter" string-parsing algorithm to instantly
    drop files that do not fall within the requested years, drastically reducing disk I/O.
    It then lazily loads the surviving NetCDF files using xarray to extract the exact
    chunk-agnostic integer indices (time slices) for multiprocessing workers.

    Args:
        main_var_dir (str): Base directory containing main variable NetCDF files.
        main_var_template (str): Filename naming convention for main variable files
            (e.g., "cerra_tp_YYYYMMDDTHHMM.nc" or "TOT_PREC_YYYY-YYYY.nc").
        env_var_dir (str, optional): Base directory containing environmental variable files.
            Defaults to None.
        env_var_template (str, optional): Filename naming convention for environmental files.
            Defaults to None.
        years (list of int, optional): Specific years to process. Files outside these years
            are filtered out. Defaults to None (process all).
        months (list of int, optional): Specific months to process. Slices outside these
            months are ignored. Defaults to None (process all).
        dt_hours (float, optional): Time step resolution in hours. Used for flooring timestamps.
            Defaults to 1.0.

    Returns:
        list of dict: A chronologically sorted list of task dictionaries required by parallel
            workers. Each dictionary contains exact file paths and NetCDF integer slice indices:
            [
                {
                    'aligned_time': Timestamp('2000-01-01 00:00:00'),
                    'main_var_file': '/path/to/main_var.nc',
                    'main_var_idx': 0,
                    'main_var_raw_time': Timestamp('2000-01-01 00:30:00'),
                    'env_var_file': '/path/to/env_var.nc',
                    'env_var_idx': 0,
                    'env_var_raw_time': Timestamp('2000-01-01 00:00:00')
                },
                ...
            ]
    """
    logger = logging.getLogger(__name__)
    tasks_dict = {}

    def _get_glob_pattern(template):
        """Converts a user template like 'file_YYYYMMDD.nc' into a glob pattern 'file_*.nc'."""
        pattern = template
        for key in ["YYYY", "MM", "DD", "HH", "mm", "ss"]:
            pattern = pattern.replace(key, "*")
        pattern = re.sub(r"\*+", "*", pattern)
        return pattern

    def scan_directory(directory, template, file_type):
        """Finds files, applies the Smart Pre-Filter, and lazily extracts metadata."""
        glob_pattern = _get_glob_pattern(template)
        search_path = os.path.join(directory, "**", glob_pattern)
        logger.info(f"Searching for {file_type} files using pattern: {search_path}")

        all_files = sorted(glob.glob(search_path, recursive=True))
        if not all_files:
            logger.warning(f"No files found for {file_type} in {directory}")
            return

        # --- SMART PRE-FILTER (Instant String Parsing) ---
        filtered_files = []
        for filepath in all_files:
            if years:
                # Strip explicit time strings (e.g., 'T2000') so they aren't confused as years
                clean_filename = re.sub(r"T\d{4}", "", os.path.basename(filepath))
                found_years = [
                    int(y) for y in re.findall(r"(19\d{2}|20\d{2})", clean_filename)
                ]

                if found_years:
                    min_y, max_y = min(found_years), max(found_years)
                    if not any(min_y <= y <= max_y for y in years):
                        continue
            filtered_files.append(filepath)

        dropped_count = len(all_files) - len(filtered_files)
        if dropped_count > 0:
            logger.info(
                f"Smart Template Filter instantly dropped {dropped_count} irrelevant {file_type} files."
            )

        logger.info(
            f"Opening metadata for remaining {len(filtered_files)} {file_type} files..."
        )

        # --- SAFE SEQUENTIAL XARRAY READ ---
        for filepath in filtered_files:
            try:
                with xr.open_dataset(filepath, engine="netcdf4") as ds:
                    if "time" not in ds:
                        logger.warning(f"No 'time' dimension in {filepath}. Skipping.")
                        continue

                    times_raw = ds["time"].values
                    times_floored = (
                        ds["time"].dt.floor(f"{int(dt_hours * 60)}min").values
                    )

                    years_arr = ds["time"].dt.year.values
                    months_arr = ds["time"].dt.month.values

                    for idx, (t, aligned_t, y, m) in enumerate(
                        zip(times_raw, times_floored, years_arr, months_arr)
                    ):
                        if years and y not in years:
                            continue
                        if months and m not in months:
                            continue

                        if aligned_t not in tasks_dict:
                            tasks_dict[aligned_t] = {"aligned_time": aligned_t}

                        tasks_dict[aligned_t][f"{file_type}_file"] = filepath
                        tasks_dict[aligned_t][f"{file_type}_idx"] = idx
                        tasks_dict[aligned_t][f"{file_type}_raw_time"] = t
            except Exception as e:
                logger.error(f"Failed to scan {filepath} for metadata: {e}")

    scan_directory(main_var_dir, main_var_template, "main_var")
    if env_var_dir and env_var_template:
        scan_directory(env_var_dir, env_var_template, "env_var")

    valid_tasks = []
    missing_env_var = 0
    missing_main_var = 0

    for t in sorted(tasks_dict.keys()):
        task = tasks_dict[t]
        has_main_var = "main_var_file" in task
        has_env_var = "env_var_file" in task

        if env_var_dir:
            if has_main_var and has_env_var:
                valid_tasks.append(task)
            elif has_main_var:
                missing_env_var += 1
            elif has_env_var:
                missing_main_var += 1
        else:
            if has_main_var:
                task["env_var_file"] = None
                task["env_var_idx"] = None
                valid_tasks.append(task)

    if missing_env_var > 0:
        logger.warning(
            f"Found {missing_env_var} timesteps with main_var but missing env_var."
        )
    if missing_main_var > 0:
        logger.info(
            f"Found {missing_main_var} timesteps with env_var but missing main_var."
        )

    logger.info(f"Total valid timesteps identified for processing: {len(valid_tasks)}")
    return valid_tasks


def get_dataset_encoding(ds):
    """
    Centralized encoding logic for all EMMA output NetCDF files.

    Ensures full compatibility with standard visualization utilities (e.g., ncview, Panoply)
    and CF metadata conventions:
    1. 1D Coordinate variables (lat, lon, time) are uncompressed (`zlib=False`).
    2. Data variables are compressed with DEFLATE (`zlib=True`, `complevel=4`).
    3. Time coordinate uses a fixed reference epoch ("days since 1950-01-01 00:00:00").
    4. Mask and ID integer variables use `_FillValue = -1` so value 0 remains background.

    Args:
        ds (xarray.Dataset): The dataset to generate encoding options for.

    Returns:
        dict: Encoding dictionary mapping variable names to xarray compression parameters.
    """
    encoding = {}

    coord_encoding = {"_FillValue": None, "zlib": False, "dtype": "float32"}

    time_encoding = {
        "_FillValue": None,
        "zlib": False,
        "dtype": "float64",
        "units": "days since 1950-01-01 00:00:00",
    }

    if "time" in ds:
        encoding["time"] = time_encoding

    for c in ["lat", "lon", "rlat", "rlon", "latitude", "longitude"]:
        if c in ds:
            encoding[c] = coord_encoding

    if "rotated_pole" in ds:
        encoding["rotated_pole"] = {"dtype": "int32"}

    int_encoding = {
        "zlib": True,
        "complevel": 4,
        "shuffle": True,
        "_FillValue": -1,
        "dtype": "int32",
    }
    float_encoding = {"zlib": True, "complevel": 4, "dtype": "float32"}
    byte_encoding = {"dtype": "int8"}

    grid_vars = [
        "final_labeled_regions",
        "env_var_regions",
        "robust_mcs_id",
        "mcs_id",
        "mcs_id_merge_split",
    ]
    for v in grid_vars:
        if v in ds:
            encoding[v] = int_encoding

    if "label_id" in ds:
        encoding["label_id"] = {"dtype": "int32"}
    if "label_lat" in ds:
        encoding["label_lat"] = float_encoding
    if "label_lon" in ds:
        encoding["label_lon"] = float_encoding

    if "active_track_id" in ds:
        encoding["active_track_id"] = {"dtype": "int32"}
    if "active_track_lat" in ds:
        encoding["active_track_lat"] = float_encoding
    if "active_track_lon" in ds:
        encoding["active_track_lon"] = float_encoding
    if "active_track_touches_boundary" in ds:
        encoding["active_track_touches_boundary"] = byte_encoding

    return encoding


def save_dataset_to_netcdf(ds, output_path):
    """
    Saves an xarray Dataset to a NetCDF file with standardized encoding options.

    Args:
        ds (xarray.Dataset): The dataset to save.
        output_path (str): File system path where the NetCDF file will be written.

    Returns:
        None
    """
    encoding = get_dataset_encoding(ds)
    ds.to_netcdf(output_path, encoding=encoding)


def handle_exception(exc_type, exc_value, exc_traceback):
    """
    Global uncaught exception handler assigned to `sys.excepthook`.

    Logs unhandled critical errors to file and console while gracefully ignoring
    user-initiated interrupts (Ctrl+C).

    Args:
        exc_type (type): Exception class type.
        exc_value (Exception): Exception instance containing error details.
        exc_traceback (traceback): Python traceback object.

    Returns:
        None
    """
    if issubclass(exc_type, KeyboardInterrupt):
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
        return

    logger = logging.getLogger(__name__)
    logger.critical("Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback))


def setup_logging(output_dir, filename="mcs_tracking.log", mode="a"):
    """
    Configures application-wide logging handlers and formatting.

    Clears pre-existing file handlers to avoid duplicate log entries and sets up
    simultaneous output to both file and standard console output.

    Args:
        output_dir (str): Directory where the log file will be saved.
        filename (str, optional): Name of the log file. Defaults to "mcs_tracking.log".
        mode (str, optional): File opening mode ('w' for overwrite, 'a' for append).
            Defaults to "a".

    Returns:
        None
    """
    log_filepath = os.path.join(output_dir, filename)
    os.makedirs(os.path.dirname(log_filepath), exist_ok=True)

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    for handler in logger.handlers[:]:
        if isinstance(handler, logging.FileHandler):
            handler.close()
            logger.removeHandler(handler)

    file_handler = logging.FileHandler(log_filepath, mode=mode)
    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    if not any(isinstance(h, logging.StreamHandler) for h in logger.handlers):
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)


def convert_main_var_units(main_var, target_unit="mm/h"):
    """
    Converts primary tracking variable values to standardized target units.

    Recognized conversions for precipitation fields:
    - 'm', 'meter', 'metre': Multiplies by 1000.0 (hourly accumulation).
    - 'kg m-2 s-1': Multiplies by 3600.0 (mm/s to mm/h conversion).
    - 'mm', 'mm/h', 'mm/hr', 'kg m-2', 'mm h-1': Leaves values unchanged (factor 1.0).

    Args:
        main_var (xarray.DataArray): DataArray containing primary tracking field.
        target_unit (str, optional): Desired unit attribute string. Defaults to "mm/h".

    Returns:
        xarray.DataArray: DataArray with converted values and updated `units` attribute.
    """
    orig_units = main_var.attrs.get("units", "").lower()

    if orig_units in ["m", "meter", "metre"]:
        factor = 1000.0
    elif orig_units in ["kg m-2 s-1"]:
        factor = 3600.0
    elif orig_units in ["mm", "mm/h", "mm/hr", "kg m-2", "mm h-1"]:
        factor = 1.0
    else:
        logger.warning(
            f"Unrecognized main_var units '{orig_units}'. No scaling applied."
        )
        factor = 1.0

    new_main_var = main_var * factor
    new_main_var.attrs["units"] = target_unit
    return new_main_var


def convert_env_var_units(env_var, target_unit="K"):
    """
    Converts environmental variable values to standardized target units.

    Recognized conversions:
    - 'K', 'Kelvin', '°C', 'degree_Celcius': Preserves difference scale offset (constant 0).

    Args:
        env_var (xarray.DataArray): DataArray containing environmental field.
        target_unit (str, optional): Desired unit attribute string. Defaults to "K".

    Returns:
        xarray.DataArray: DataArray with updated `units` attribute.
    """
    orig_units = env_var.attrs.get("units", "")

    if orig_units in ["K", "Kelvin", "°C", "degree_Celcius"]:
        constant = 0
    else:
        logger.warning(f"Unrecognized env_var units '{orig_units}'. No offset applied.")
        constant = 0

    new_env_var = env_var + constant
    new_env_var.attrs["units"] = target_unit
    return new_env_var


def load_main_var_data(file_path, data_var, y_dim_name, x_dim_name, time_index=0):
    """
    Loads a dataset timestep and extracts converted primary variable values and spatial coordinates.

    Implements strict grid validation:
    1. Reads 1D native spatial dimensions (`y_dim_name`, `x_dim_name`).
    2. Searches for 2D auxiliary geographic coordinates (`latitude`/`longitude`, `lat`/`lon`).
    3. Raises `ValueError` if rotated grid dimensions exist without auxiliary 2D coordinates.

    Args:
        file_path (str): Path to the NetCDF file.
        data_var (str): Variable name of the primary tracking field.
        y_dim_name (str): Latitude / y-dimension variable name.
        x_dim_name (str): Longitude / x-dimension variable name.
        time_index (int, optional): NetCDF integer slice index along the time dimension.
            Defaults to 0.

    Returns:
        tuple: (ds, lat2d, lon2d, native_y, native_x, main_var_converted)
            - ds (xarray.Dataset): Sliced dataset at `time_index`.
            - lat2d (numpy.ndarray): 2D array of true geographic latitudes.
            - lon2d (numpy.ndarray): 2D array of true geographic longitudes.
            - native_y (numpy.ndarray): 1D array of native y-coordinates.
            - native_x (numpy.ndarray): 1D array of native x-coordinates.
            - main_var_converted (xarray.DataArray): 2D unit-converted main variable values.

    Raises:
        ValueError: If native grid configuration is invalid or missing auxiliary 2D coordinates.
    """
    ds = xr.open_dataset(file_path, engine="netcdf4")
    ds = ds.isel(time=time_index)

    native_y = ds[y_dim_name].values
    native_x = ds[x_dim_name].values

    lat2d, lon2d = None, None

    if native_y.ndim == 1 and native_x.ndim == 1:
        aux_candidates = [("lat", "lon"), ("latitude", "longitude")]

        for aux_lat, aux_lon in aux_candidates:
            if aux_lat in ds and aux_lon in ds:
                expected_shape = (len(native_y), len(native_x))
                if ds[aux_lat].ndim == 2 and ds[aux_lat].shape == expected_shape:
                    lat2d = ds[aux_lat].values
                    lon2d = ds[aux_lon].values
                    break

        if lat2d is None:
            is_rotated_dim = "rlat" in y_dim_name or "rlon" in x_dim_name
            if is_rotated_dim:
                raise ValueError(
                    f"STRICT MODE ERROR: Native dimensions are '{y_dim_name}'/'{x_dim_name}', "
                    "implying a rotated grid. However, no valid 2D geographic coordinates "
                    "were found. Aborting to prevent georeferencing errors."
                )
            lon2d, lat2d = np.meshgrid(native_x, native_y)
    else:
        raise ValueError("Please provide the name of 1D dimension coordinates.")

    main_var_da = ds[str(data_var)]
    main_var_converted = convert_main_var_units(main_var_da)

    return ds, lat2d, lon2d, native_y, native_x, main_var_converted


def load_env_var_data(file_path, data_var, y_dim_name, x_dim_name, time_index=0):
    """
    Loads a dataset timestep and extracts environmental variable values and spatial coordinates.

    Implements strict grid validation identical to `load_main_var_data`.

    Args:
        file_path (str): Path to the NetCDF file.
        data_var (str): Variable name of the environmental field.
        y_dim_name (str): Latitude / y-dimension variable name.
        x_dim_name (str): Longitude / x-dimension variable name.
        time_index (int, optional): NetCDF integer slice index along the time dimension.
            Defaults to 0.

    Returns:
        tuple: (ds, lat2d, lon2d, native_y, native_x, env_var_converted)
            - ds (xarray.Dataset): Sliced dataset at `time_index` with unused variables dropped.
            - lat2d (numpy.ndarray): 2D array of true geographic latitudes.
            - lon2d (numpy.ndarray): 2D array of true geographic longitudes.
            - native_y (numpy.ndarray): 1D array of native y-coordinates.
            - native_x (numpy.ndarray): 1D array of native x-coordinates.
            - env_var_converted (xarray.DataArray): 2D unit-converted environmental values.

    Raises:
        ValueError: If native grid configuration is invalid or missing auxiliary 2D coordinates.
    """
    ds = xr.open_dataset(file_path, engine="netcdf4")
    ds = ds.isel(time=time_index)

    native_y = ds[y_dim_name].values
    native_x = ds[x_dim_name].values

    lat2d, lon2d = None, None

    if native_y.ndim == 1 and native_x.ndim == 1:
        aux_candidates = [("lat", "lon"), ("latitude", "longitude")]

        for aux_lat, aux_lon in aux_candidates:
            if aux_lat in ds and aux_lon in ds:
                expected_shape = (len(native_y), len(native_x))
                if ds[aux_lat].ndim == 2 and ds[aux_lat].shape == expected_shape:
                    lat2d = ds[aux_lat].values
                    lon2d = ds[aux_lon].values
                    break

        if lat2d is None:
            is_rotated_dim = "rlat" in y_dim_name or "rlon" in x_dim_name
            if is_rotated_dim:
                raise ValueError(
                    f"STRICT MODE ERROR: Native dimensions are '{y_dim_name}'/'{x_dim_name}', "
                    "implying a rotated grid but 2D coordinates are missing."
                )
            lon2d, lat2d = np.meshgrid(native_x, native_y)
    else:
        raise ValueError("Please provide the name of 1D dimension coordinates.")

    env_var_da = ds[str(data_var)]
    env_var_converted = convert_env_var_units(env_var_da, target_unit="K")

    data_vars_list = [v for v in ds.data_vars]
    if data_var in data_vars_list:
        data_vars_list.remove(data_var)
    ds = ds.drop_vars(data_vars_list, errors="ignore")

    return ds, lat2d, lon2d, native_y, native_x, env_var_converted


def serialize_center_points(center_points):
    """
    Serializes cluster centroid coordinates dictionary to a JSON string.

    Converts single-precision floating-point types (`float32`) to native Python floats.

    Args:
        center_points (dict): Mapping `{label_id_str: (lat_val, lon_val)}`.

    Returns:
        str: JSON-encoded string representation of cluster center points.
    """
    casted_dict = {}
    for label_str, (lat_val, lon_val) in center_points.items():
        casted_dict[label_str] = (float(lat_val), float(lon_val))
    return json.dumps(casted_dict)


def load_individual_detection_files(
    year_input_dir, use_env_filter, y_dim_name, x_dim_name
):
    """
    Loads a sequence of hourly detection result NetCDF files for a given year.

    Builds spatial grid coordinates and template information once from the first file
    to save memory across large annual time series.

    Args:
        year_input_dir (str): Directory path containing detection NetCDF files for a year.
        use_env_filter (bool): If True, loads 2D environmental mask arrays.
        y_dim_name (str): Variable name of 1D y-dimension.
        x_dim_name (str): Variable name of 1D x-dimension.

    Returns:
        tuple: (detection_results_list, grid_info)
            - detection_results_list (list of dict): Chronologically sorted dictionaries
              containing frame timestamp, labeled mask, center points, and environmental regions.
            - grid_info (dict): Static global spatial grid template.
    """
    detection_results = []
    grid_info = None

    file_pattern = os.path.join(year_input_dir, "**", "detection_*.nc")
    filepaths = sorted(glob.glob(file_pattern, recursive=True))

    if not filepaths:
        logger.warning(f"No detection files found matching {file_pattern}")
        return [], None

    for filepath in filepaths:
        try:
            with xr.open_dataset(filepath, engine="netcdf4") as ds:
                time_val = ds["time"].values[0]

                if grid_info is None:
                    lat_1d = ds[y_dim_name].values
                    lon_1d = ds[x_dim_name].values

                    lat2d = ds["latitude"].values
                    lon2d = ds["longitude"].values

                    grid_info = build_grid_info(
                        ds, y_dim_name, x_dim_name, lat2d, lon2d
                    )

                final_labeled_regions = ds["final_labeled_regions"].values[0]

                center_points_dict = {}
                if "label_id" in ds:
                    ids = ds["label_id"].values
                    lats = ds["label_lat"].values
                    lons = ds["label_lon"].values
                    for i, label_id in enumerate(ids):
                        lbl_str = str(int(label_id))
                        lbl_lat = float(lats[i])
                        lbl_lon = float(lons[i])
                        if np.isnan(lbl_lat) or np.isnan(lbl_lon):
                            center_points_dict[lbl_str] = None
                        else:
                            center_points_dict[lbl_str] = (lbl_lat, lbl_lon)
                elif "center_points_t0" in ds.attrs:
                    try:
                        center_points_dict = json.loads(ds.attrs["center_points_t0"])
                        if isinstance(center_points_dict, str):
                            center_points_dict = json.loads(center_points_dict)
                    except Exception:
                        center_points_dict = {}

                detection_result = {
                    "final_labeled_regions": final_labeled_regions,
                    "time": time_val,
                    "center_points": center_points_dict,
                }

                if use_env_filter:
                    if "env_var_regions" in ds:
                        detection_result["env_var_regions"] = ds[
                            "env_var_regions"
                        ].values[0]
                    else:
                        detection_result["env_var_regions"] = np.zeros_like(
                            final_labeled_regions
                        )

                detection_results.append(detection_result)

        except Exception as e:
            logger.error(f"Error loading {filepath}: {e}")
            continue

    detection_results.sort(key=lambda x: x["time"])
    return detection_results, grid_info


def save_detection_result(
    detection_result, output_dir, data_source, grid_info, config=None
):
    """
    Saves a single timestep's detection result to a compressed, CF-compliant NetCDF file.

    Formats output into `{output_dir}/YYYY/MM/detection_YYYYMMDDTHHMM.nc`. Attaches active
    configuration parameters (omitting disabled options) and dynamic variable unit metadata.

    Args:
        detection_result (dict): Detection frame data containing:
            - 'time': Datetime timestamp.
            - 'final_labeled_regions': 2D integer array of detected object labels.
            - 'env_var_regions': 2D binary environmental mask.
            - 'center_points': Dict mapping label ID string to (lat, lon).
        output_dir (str): Base root directory where subfolder output tree is created.
        data_source (str): Text description of input dataset.
        grid_info (dict): Global verified grid template.
        config (EmmaConfig, optional): EmmaConfig object used to append active top-level run metadata.

    Returns:
        None
    """
    time_raw = detection_result["time"]
    try:
        time_obj = pd.to_datetime(time_raw).round("s")
    except (TypeError, ValueError):
        time_obj = time_raw.item() if hasattr(time_raw, "item") else time_raw

    year_str = time_obj.strftime("%Y")
    month_str = time_obj.strftime("%m")

    structured_dir = os.path.join(output_dir, year_str, month_str)
    os.makedirs(structured_dir, exist_ok=True)

    filename = f"detection_{time_obj.strftime('%Y%m%dT%H%M')}.nc"
    output_filepath = os.path.join(structured_dir, filename)

    y_dim = grid_info["y_dim_name"]
    x_dim = grid_info["x_dim_name"]
    y_1d = grid_info["lat1d"]
    x_1d = grid_info["lon1d"]

    center_points = detection_result.get("center_points", {})
    if center_points:
        sorted_labels = sorted(center_points.keys(), key=lambda x: int(x))
        label_ids = np.array([int(lbl) for lbl in sorted_labels], dtype=np.int32)
        label_lats = [
            center_points[lbl][0] if center_points[lbl] else np.nan
            for lbl in sorted_labels
        ]
        label_lons = [
            center_points[lbl][1] if center_points[lbl] else np.nan
            for lbl in sorted_labels
        ]
    else:
        label_ids = np.array([], dtype=np.int32)
        label_lats = []
        label_lons = []

    final_labeled_regions = np.expand_dims(
        detection_result["final_labeled_regions"], axis=0
    )

    env_mask = detection_result.get(
        "env_var_regions",
        detection_result.get(
            "env_var_regions", np.zeros_like(detection_result["final_labeled_regions"])
        ),
    )
    env_var_regions = np.expand_dims(env_mask, axis=0)

    data_vars = {
        "final_labeled_regions": (["time", y_dim, x_dim], final_labeled_regions),
        "env_var_regions": (["time", y_dim, x_dim], env_var_regions),
        "label_id": (["labels"], label_ids),
        "label_lat": (["labels"], label_lats),
        "label_lon": (["labels"], label_lons),
    }

    # Explicitly include time as both coordinate and dimension mapping
    ds = xr.Dataset(
        data_vars=data_vars,
        coords={
            "time": ("time", [pd.Timestamp(time_obj)]),
            y_dim: y_1d,
            x_dim: x_1d,
        },
    )

    ds["latitude"] = ((y_dim, x_dim), grid_info["lat2d"])
    ds["longitude"] = ((y_dim, x_dim), grid_info["lon2d"])

    ds.attrs = {
        "title": "EMMA-Tracker Detection Output",
        "institution": "Wegener Center for Climate and Global Change, University of Graz",
        "source": data_source,
        "history": f"Created on {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "references": "Kneidinger et al. (2025)",
        "Conventions": "CF-1.7",
        "project": "EMMA",
        "main_var_units": grid_info.get("main_var_units", "unknown"),
    }

    if grid_info.get("env_var_units") and grid_info["env_var_units"] != "unknown":
        ds.attrs["env_var_units"] = grid_info["env_var_units"]

    if config and hasattr(config, "get_active_metadata"):
        for key, val in config.get_active_metadata().items():
            ds.attrs[key] = val

    ds["time"].attrs = {"standard_name": "time"}
    ds["latitude"].attrs = {"standard_name": "latitude", "units": "degrees_north"}
    ds["longitude"].attrs = {"standard_name": "longitude", "units": "degrees_east"}

    cf_meta = grid_info.get("cf_metadata", {})
    mapping_name = cf_meta.get("grid_mapping_name", "crs")

    ds[y_dim].attrs = {}
    ds[x_dim].attrs = {}

    if mapping_name == "rotated_latitude_longitude":
        ds[y_dim].attrs["standard_name"] = "grid_latitude"
        ds[y_dim].attrs["units"] = "degrees"
        ds[x_dim].attrs["standard_name"] = "grid_longitude"
        ds[x_dim].attrs["units"] = "degrees"
        var_name = "rotated_pole"

        ds[var_name] = ([], np.int32(0))
        ds[var_name].attrs = {
            "grid_mapping_name": "rotated_latitude_longitude",
            "grid_north_pole_latitude": float(
                cf_meta.get("grid_north_pole_latitude", 39.25)
            ),
            "grid_north_pole_longitude": float(
                cf_meta.get("grid_north_pole_longitude", -162.0)
            ),
        }
    else:
        ds[y_dim].attrs["standard_name"] = "latitude"
        ds[y_dim].attrs["units"] = "degrees_north"
        ds[x_dim].attrs["standard_name"] = "longitude"
        ds[x_dim].attrs["units"] = "degrees_east"
        var_name = "crs"
        ds[var_name] = ([], np.int32(0))
        ds[var_name].attrs = {"grid_mapping_name": mapping_name}

    for var in ["final_labeled_regions", "env_var_regions"]:
        if var in ds:
            ds[var].attrs["grid_mapping"] = var_name
            ds[var].attrs["coordinates"] = "latitude longitude"
            ds[var].attrs["cell_methods"] = "time: point"

    ds["final_labeled_regions"].attrs.update(
        {"long_name": "Labeled Convective Regions", "units": "1"}
    )
    ds["env_var_regions"].attrs.update(
        {"long_name": "Environmental Mask", "units": "1"}
    )
    ds["label_id"].attrs.update({"long_name": "Feature Label IDs"})

    save_dataset_to_netcdf(ds, output_filepath)


def save_tracking_result(
    tracking_data_for_timestep, output_dir, data_source, grid_info, config=None
):
    """
    Saves a single timestep's tracking results to a compressed, CF-compliant NetCDF file.

    Stores both 2D segmentation masks (`robust_mcs_id`, `mcs_id`, `mcs_id_merge_split`) and tabular
    per-frame summary arrays (`active_track_id`, `active_track_lat`, `active_track_lon`, boundary flags).
    Appends active run metadata and dynamic units to global dataset attributes.

    Args:
        tracking_data_for_timestep (dict): Tracking frame dictionary containing:
            - 'time': Datetime timestamp.
            - 'robust_mcs_id': 2D integer array of robust/mature phase track IDs.
            - 'mcs_id': 2D integer array of main lifecycle track IDs.
            - 'mcs_id_merge_split': 2D integer array including merger/split family history.
            - 'tracking_centers': Dict mapping track ID string to (lat, lon) center coordinates.
        output_dir (str): Root destination directory for structured output.
        data_source (str): Text string describing source data.
        grid_info (dict): Global verified grid template.
        config (EmmaConfig, optional): EmmaConfig object used to append active top-level run metadata.

    Returns:
        None
    """
    time_raw = tracking_data_for_timestep["time"]
    try:
        time_obj = pd.to_datetime(time_raw).round("s")
    except (TypeError, ValueError):
        time_obj = time_raw.item() if hasattr(time_raw, "item") else time_raw

    year_str = time_obj.strftime("%Y")
    month_str = time_obj.strftime("%m")

    structured_dir = os.path.join(output_dir, year_str, month_str)
    os.makedirs(structured_dir, exist_ok=True)

    filename = f"tracking_{time_obj.strftime('%Y%m%dT%H%M')}.nc"
    output_filepath = os.path.join(structured_dir, filename)

    y_dim = grid_info["y_dim_name"]
    x_dim = grid_info["x_dim_name"]
    y_1d = grid_info["lat1d"]
    x_1d = grid_info["lon1d"]

    centers_dict = tracking_data_for_timestep.get("tracking_centers", {})
    grid = tracking_data_for_timestep["mcs_id"]
    if grid.ndim == 3:
        grid = grid[0]
    ymax, xmax = grid.shape[0] - 1, grid.shape[1] - 1

    active_ids = []
    active_lats = []
    active_lons = []
    active_boundary_flags = []

    if centers_dict:
        sorted_ids = sorted(centers_dict.keys(), key=lambda x: int(x))
        active_ids = np.array([int(tid) for tid in sorted_ids], dtype=np.int32)

        for tid in sorted_ids:
            coords = centers_dict[tid]
            if coords and coords[0] is not None:
                active_lats.append(coords[0])
                active_lons.append(coords[1])
            else:
                active_lats.append(np.nan)
                active_lons.append(np.nan)

            mask = grid == int(tid)
            touches = (
                np.any(mask[0, :])
                or np.any(mask[ymax, :])
                or np.any(mask[:, 0])
                or np.any(mask[:, xmax])
            )
            active_boundary_flags.append(int(touches))

        active_boundary_flags = np.array(active_boundary_flags, dtype=np.int8)
    else:
        active_ids = np.array([], dtype=np.int32)
        active_lats = []
        active_lons = []
        active_boundary_flags = np.array([], dtype=np.int8)

    robust_mcs_id_arr = np.expand_dims(
        tracking_data_for_timestep["robust_mcs_id"], axis=0
    )
    mcs_id_arr = np.expand_dims(tracking_data_for_timestep["mcs_id"], axis=0)
    mcs_id_merge_split_arr = np.expand_dims(
        tracking_data_for_timestep["mcs_id_merge_split"], axis=0
    )

    data_vars = {
        "robust_mcs_id": (["time", y_dim, x_dim], robust_mcs_id_arr),
        "mcs_id": (["time", y_dim, x_dim], mcs_id_arr),
        "mcs_id_merge_split": (["time", y_dim, x_dim], mcs_id_merge_split_arr),
        "active_track_id": (["tracks"], active_ids),
        "active_track_lat": (["tracks"], active_lats),
        "active_track_lon": (["tracks"], active_lons),
        "active_track_touches_boundary": (["tracks"], active_boundary_flags),
    }

    ds = xr.Dataset(
        data_vars=data_vars,
        coords={
            "time": [time_obj],
            y_dim: y_1d,
            x_dim: x_1d,
        },
    )

    ds["latitude"] = ((y_dim, x_dim), grid_info["lat2d"])
    ds["longitude"] = ((y_dim, x_dim), grid_info["lon2d"])

    ds.attrs = {
        "title": "EMMA-Tracker Output",
        "institution": "Wegener Center for Climate and Global Change, University of Graz",
        "source": data_source,
        "history": f"Created on {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "Conventions": "CF-1.7",
        "project": "EMMA",
        "main_var_units": grid_info.get("main_var_units", "unknown"),
    }

    if grid_info.get("env_var_units") and grid_info["env_var_units"] != "unknown":
        ds.attrs["env_var_units"] = grid_info["env_var_units"]

    if config and hasattr(config, "get_active_metadata"):
        for key, val in config.get_active_metadata().items():
            ds.attrs[key] = val

    ds["time"].attrs = {"standard_name": "time"}
    ds["latitude"].attrs = {"standard_name": "latitude", "units": "degrees_north"}
    ds["longitude"].attrs = {"standard_name": "longitude", "units": "degrees_east"}

    cf_meta = grid_info.get("cf_metadata", {})
    mapping_name = cf_meta.get("grid_mapping_name", "crs")

    ds[y_dim].attrs = {}
    ds[x_dim].attrs = {}

    if mapping_name == "rotated_latitude_longitude":
        ds[y_dim].attrs["standard_name"] = "grid_latitude"
        ds[y_dim].attrs["units"] = "degrees"
        ds[x_dim].attrs["standard_name"] = "grid_longitude"
        ds[x_dim].attrs["units"] = "degrees"
        var_name = "rotated_pole"

        ds[var_name] = ([], np.int32(0))
        ds[var_name].attrs = {
            "grid_mapping_name": "rotated_latitude_longitude",
            "grid_north_pole_latitude": float(
                cf_meta.get("grid_north_pole_latitude", 39.25)
            ),
            "grid_north_pole_longitude": float(
                cf_meta.get("grid_north_pole_longitude", -162.0)
            ),
        }
    else:
        ds[y_dim].attrs["standard_name"] = "latitude"
        ds[y_dim].attrs["units"] = "degrees_north"
        ds[x_dim].attrs["standard_name"] = "longitude"
        ds[x_dim].attrs["units"] = "degrees_east"
        var_name = "crs"
        ds[var_name] = ([], np.int32(0))
        ds[var_name].attrs = {"grid_mapping_name": mapping_name}

    grid_vars = ["robust_mcs_id", "mcs_id", "mcs_id_merge_split"]
    for var in grid_vars:
        if var in ds:
            ds[var].attrs["grid_mapping"] = var_name
            ds[var].attrs["coordinates"] = "latitude longitude"
            ds[var].attrs["cell_methods"] = "time: point"

    ds["robust_mcs_id"].attrs.update(
        {"long_name": "Robust Mature MCS Track IDs", "units": "1"}
    )
    ds["mcs_id"].attrs.update({"long_name": "Main MCS Track IDs", "units": "1"})
    ds["mcs_id_merge_split"].attrs.update(
        {"long_name": "Family Tree Track IDs", "units": "1"}
    )
    ds["active_track_id"].attrs = {"long_name": "Active Track IDs"}
    ds["active_track_touches_boundary"].attrs = {
        "long_name": "Boundary Touching Flag",
        "units": "1",
    }

    save_dataset_to_netcdf(ds, output_filepath)
