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
    chunk-agnostic integer indices (time slices) for the multiprocessing workers.

    Args:
        main_var_dir (str): Base directory containing main variable NetCDF files.
        main_var_template (str): Filename naming convention for main variable files
            (e.g., "cerra_tp_YYYYMMDDTHHMM.nc" or "TOT_PREC_YYYY-YYYY.nc").
        env_var_dir (str, optional): Base directory containing environmental variable files. Defaults to None.
        env_var_template (str, optional): Filename naming convention for env files. Defaults to None.
        years (list of int, optional): Specific years to process. Files outside these years
            are filtered out. Defaults to None (process all).
        months (list of int, optional): Specific months to process. Slices outside these
            months are ignored. Defaults to None (process all).
        dt_hours (float, optional): Time step resolution in hours. Used for flooring timestamps.

    Returns:
        list of dict: A chronologically sorted list of dictionaries. Each dictionary represents
            a single valid hourly timestep (a "task") containing the exact file paths and
            NetCDF integer slice indices required by the parallel workers.
            Example:
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
        # Replace common datetime placeholders with asterisks
        for key in ["YYYY", "MM", "DD", "HH", "mm", "ss"]:
            pattern = pattern.replace(key, "*")
        # Collapse multiple asterisks into a single one for cleaner globbing
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
                # Strip out explicit time strings (like 'T2000') so they aren't confused as years
                clean_filename = re.sub(r"T\d{4}", "", os.path.basename(filepath))
                # Find all 4-digit sequences that look like years (1900-2099)
                found_years = [
                    int(y) for y in re.findall(r"(19\d{2}|20\d{2})", clean_filename)
                ]

                if found_years:
                    min_y, max_y = min(found_years), max(found_years)
                    # If the requested years do not overlap with the file's year bounds at all, skip it
                    if not any(min_y <= y <= max_y for y in years):
                        continue
            filtered_files.append(filepath)

        dropped_count = len(all_files) - len(filtered_files)
        if dropped_count > 0:
            logger.info(
                f"Smart Template Filter instantly dropped {dropped_count} irrelevant {file_type} files."
            )

        logger.info(
            f"Opening metadata for the remaining {len(filtered_files)} {file_type} files (Sequential)..."
        )

        # --- SAFE SEQUENTIAL XARRAY READ ---
        for filepath in filtered_files:
            try:
                # Lazy load: only reads metadata headers, avoids HDF5 threading crashes
                with xr.open_dataset(filepath, engine="netcdf4") as ds:
                    if "time" not in ds:
                        logger.warning(f"No 'time' dimension in {filepath}. Skipping.")
                        continue

                    times_raw = ds["time"].values
                    times_floored = ds["time"].dt.floor(f"{int(dt_hours * 60)}min").values

                    # Safely extract years and months using xarray's dt accessor
                    years_arr = ds["time"].dt.year.values
                    months_arr = ds["time"].dt.month.values

                    for idx, (t, aligned_t, y, m) in enumerate(
                        zip(times_raw, times_floored, years_arr, months_arr)
                    ):
                        # Sub-filter indices within the file based on requested config
                        if years and y not in years:
                            continue
                        if months and m not in months:
                            continue

                        # Initialize alignment key if it doesn't exist
                        if aligned_t not in tasks_dict:
                            tasks_dict[aligned_t] = {"aligned_time": aligned_t}

                        tasks_dict[aligned_t][f"{file_type}_file"] = filepath
                        tasks_dict[aligned_t][f"{file_type}_idx"] = idx
                        tasks_dict[aligned_t][f"{file_type}_raw_time"] = t
            except Exception as e:
                logger.error(f"Failed to scan {filepath} for metadata: {e}")

    # Execute Scans for both directories
    scan_directory(main_var_dir, main_var_template, "main_var")
    if env_var_dir and env_var_template:
        scan_directory(env_var_dir, env_var_template, "env_var")

    # Build Final Tasks List by checking for complete pairs
    valid_tasks = []
    missing_env_var = 0
    missing_main_var = 0

    for t in sorted(tasks_dict.keys()):
        task = tasks_dict[t]
        has_main_var = "main_var_file" in task
        has_env_var = "env_var_file" in task

        if env_var_dir:
            # Both files are required for a valid task
            if has_main_var and has_env_var:
                valid_tasks.append(task)
            elif has_main_var:
                missing_env_var += 1
            elif has_env_var:
                missing_main_var += 1
        else:
            # Only main_var is required
            if has_main_var:
                task["env_var_file"] = None
                task["env_var_idx"] = None
                valid_tasks.append(task)

    if missing_env_var > 0:
        logger.warning(f"Found {missing_env_var} timesteps with main_var but missing env_var.")
    if missing_main_var > 0:
        logger.info(f"Found {missing_main_var} timesteps with env_var but missing main_var.")

    logger.info(f"Total valid timesteps identified for processing: {len(valid_tasks)}")
    return valid_tasks


def get_dataset_encoding(ds):
    """
    Centralized encoding logic for all EMMA output files.
    Ensures consistency across detection, tracking, and post-processing.

    Fixes for ncview:
    1. 1D Coordinates (lat, lon, time) are UNCOMPRESSED (zlib=False).
    2. Data variables are COMPRESSED (zlib=True).
    3. Time uses a fixed epoch.
    4. Masks use _FillValue = -1 so 0 is visible as background.
    """
    encoding = {}

    # --- 1. Coordinate Encoding (Uncompressed, No Fill Value) ---
    coord_encoding = {"_FillValue": None, "zlib": False, "dtype": "float32"}

    # Standardize Time Epoch
    time_encoding = {
        "_FillValue": None,
        "zlib": False,
        "dtype": "float64",
        "units": "days since 1950-01-01 00:00:00",
    }

    # --- 2. Data Encoding (Compressed) ---
    int_encoding = {
        "zlib": True,
        "complevel": 4,
        "shuffle": True,
        "_FillValue": -1,  # Critical: 0 becomes valid background
        "dtype": "int32",
    }
    float_encoding = {"zlib": True, "complevel": 4, "dtype": "float32"}
    byte_encoding = {"dtype": "int8"}

    # Apply Rules based on Variable Existence
    if "time" in ds:
        encoding["time"] = time_encoding

    # 1D Coordinates
    for c in ["lat", "lon", "rlat", "rlon"]:
        if c in ds:
            encoding[c] = coord_encoding

    # 2D Coordinates (Auxiliary) -> Compressed to save space
    for c in ["latitude", "longitude"]:
        if c in ds:
            encoding[c] = float_encoding

    # Rotated Pole Container
    if "rotated_pole" in ds:
        encoding["rotated_pole"] = {"dtype": "int32"}

    # Gridded Data Variables
    grid_vars = [
        "final_labeled_regions",
        "lifted_index_regions",
        "robust_mcs_id",
        "mcs_id",
        "mcs_id_merge_split",
    ]
    for v in grid_vars:
        if v in ds:
            encoding[v] = int_encoding

    # Tabular (Track) Variables
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
    Shared function to save any EMMA Xarray Dataset to NetCDF.
    Applies the standardized encoding and compression rules.
    """
    encoding = get_dataset_encoding(ds)
    ds.to_netcdf(output_path, encoding=encoding)


def handle_exception(exc_type, exc_value, exc_traceback):
    """
    Global exception handler to log any uncaught exceptions.
    This is assigned to sys.excepthook in main.py.
    """
    # Don't log KeyboardInterrupt (Ctrl+C) as a critical error
    if issubclass(exc_type, KeyboardInterrupt):
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
        return

    logger = logging.getLogger(__name__)
    logger.critical("Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback))


def setup_logging(output_dir, filename="mcs_tracking.log", mode="a"):
    """
    Configures logging. Removes old handlers to prevent duplicate messages.
    """
    log_filepath = os.path.join(output_dir, filename)
    os.makedirs(os.path.dirname(log_filepath), exist_ok=True)

    logger = logging.getLogger()  # Get the root logger
    logger.setLevel(logging.INFO)

    # Shut down and remove existing file handlers
    for handler in logger.handlers[:]:
        if isinstance(handler, logging.FileHandler):
            handler.close()
            logger.removeHandler(handler)

    # Add the new file handler
    file_handler = logging.FileHandler(log_filepath, mode=mode)
    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Ensure console output is still active
    if not any(isinstance(h, logging.StreamHandler) for h in logger.handlers):
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)


def convert_main_var_units(main_var, target_unit="mm/h"):
    """
    Convert the main variable DataArray to the target unit if required.

    Recognized unit conversions for precipitation:
      - 'm', 'meter', 'metre': multiply by 1000 (assumed hourly accumulation)
      - 'kg m-2 s-1': multiply by 3600 (from mm/s to mm/h, given 1 kg/m² = 1 mm water)
      - 'mm', 'mm/h', 'mm/hr': no conversion needed

    Parameters:
    - main_var: xarray DataArray of main variable values.
    - target_unit: Desired unit for the output (default: "mm/h").

    Returns:
    - new_main_var: DataArray with converted values and updated units attribute.
    """
    orig_units = main_var.attrs.get("units", "").lower()

    if orig_units in ["m", "meter", "metre"]:
        factor = 1000.0
    elif orig_units in ["kg m-2 s-1"]:
        factor = 3600.0
    elif orig_units in ["mm", "mm/h", "mm/hr", "kg m-2", "mm h-1"]:
        factor = 1.0
    else:
        print(
            f"Warning: Unrecognized main_var units '{orig_units}'. No conversion applied."
        )
        factor = 1.0

    new_main_var = main_var * factor
    new_main_var.attrs["units"] = target_unit
    return new_main_var


def convert_env_var_units(env_var, target_unit="K"):
    """
    Convert the environmental variable DataArray to the target unit if required.

    Recognized unit conversions:
      - degree Celsius to K

    Parameters:
    - env_var: xarray DataArray of environmental variable values.
    - target_unit: Desired unit for the output (default: "K").

    Returns:
    - new_env_var: DataArray with converted values and updated units attribute.
    """
    orig_units = env_var.attrs.get("units", "")

    if orig_units in ["K", "Kelvin"]:
        constant = 0
    elif orig_units in ["°C", "degree_Celcius"]:
        constant = 0  # Lifted index / temperature difference measures retain offset
    else:
        print(
            f"Warning: Unrecognized env_var units '{orig_units}'. No conversion applied."
        )
        constant = 0  # Default to 0 to be safe

    new_env_var = env_var + constant
    new_env_var.attrs["units"] = target_unit
    return new_env_var


def load_main_var_data(file_path, data_var, y_dim_name, x_dim_name, time_index=0):
    """
    Load the dataset and select the specified time step, scaling the main variable
    to target units for consistency with detection thresholds.

    This function implements STRICT grid validation:
    1. It loads the native grid coordinates (1D or 2D) based on y_dim_name/x_dim_name.
    2. It explicitly searches for 2D auxiliary geographic coordinates (lat/lon) if the
       native grid is 1D (e.g. CORDEX rotated grids).
    3. It raises a ValueError if rotated dimensions are detected but no 2D geographic
       coordinates are found, preventing silent georeferencing errors.

    Parameters:
    - file_path: Path to the NetCDF file.
    - data_var: Name of the main variable.
    - y_dim_name: Name of the latitude variable (native dimension).
    - x_dim_name: Name of the longitude variable (native dimension).
    - time_index: Index of the time step to select.

    Returns:
    - ds: xarray Dataset for the selected time.
    - lat2d: 2D array of latitudes (True Geographic Coordinates).
    - lon2d: 2D array of longitudes (True Geographic Coordinates).
    - lat: 1D array of native latitudes (or y-indices).
    - lon: 1D array of native longitudes (or x-indices).
    - main_var_converted: 2D DataArray of converted main variable values.
    """
    ds = xr.open_dataset(file_path, engine="netcdf4")
    ds = ds.isel(time=time_index)

    # 1. Load Native 1D Dimensions
    native_y = ds[y_dim_name].values
    native_x = ds[x_dim_name].values

    lat2d, lon2d = None, None

    # 2. Determine 2D Geographic Coordinates
    if native_y.ndim == 1 and native_x.ndim == 1:
        aux_candidates = [("lat", "lon"), ("latitude", "longitude")]

        for aux_lat, aux_lon in aux_candidates:
            if aux_lat in ds and aux_lon in ds:
                expected_shape = (len(native_y), len(native_x))
                if ds[aux_lat].ndim == 2 and ds[aux_lat].shape == expected_shape:
                    lat2d = ds[aux_lat].values
                    lon2d = ds[aux_lon].values
                    break

        # 3. Strict Decision Logic
        if lat2d is None:
            is_rotated_dim = "rlat" in y_dim_name or "rlon" in x_dim_name
            if is_rotated_dim:
                raise ValueError(
                    f"STRICT MODE ERROR: Native dimensions are '{y_dim_name}'/'{x_dim_name}', "
                    "implying a rotated grid. However, no valid 2D geographic coordinates "
                    "were found. Aborting to prevent georeferencing errors."
                )
            # Create meshgrid for regular grids
            lon2d, lat2d = np.meshgrid(native_x, native_y)

    else:
        raise ValueError("Please provide the name of the 1D dimension coordinates.")

    main_var_da = ds[str(data_var)]
    main_var_converted = convert_main_var_units(main_var_da)

    return ds, lat2d, lon2d, native_y, native_x, main_var_converted


def load_env_var_data(file_path, data_var, y_dim_name, x_dim_name, time_index=0):
    """
    Load the dataset and select the specified time step, scaling the environmental
    variable DataArray to target units for consistency with detection thresholds.

    This function implements STRICT grid validation identical to load_main_var_data:
    1. Loads native coordinates.
    2. Searches for 2D auxiliary coordinates if native coords are 1D.
    3. Raises ValueError if rotated grid implied but 2D coords missing.

    Parameters:
    - file_path: Path to the NetCDF file.
    - data_var: Name of the environmental variable.
    - y_dim_name: Name of the latitude variable.
    - x_dim_name: Name of the longitude variable.
    - time_index: Index of the time step to select.

    Returns:
    - ds: xarray Dataset for the selected time.
    - lat2d: 2D array of latitudes (True Geographic Coordinates).
    - lon2d: 2D array of longitudes (True Geographic Coordinates).
    - lat: 1D array of native latitudes.
    - lon: 1D array of native longitudes
    - env_var_converted: 2D DataArray of environmental variable values (scaled to K).
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
        raise ValueError("Please provide the name of the 1D dimension coordinates.")

    env_var_da = ds[str(data_var)]
    env_var_converted = convert_env_var_units(env_var_da, target_unit="K")

    # Drop unused vars to save memory
    data_vars_list = [v for v in ds.data_vars]
    if data_var in data_vars_list:
        data_vars_list.remove(data_var)
    ds = ds.drop_vars(data_vars_list, errors="ignore")

    return ds, lat2d, lon2d, native_y, native_x, env_var_converted


def serialize_center_points(center_points):
    """Convert a center_points dict with float32 lat/lon to Python floats so json.dumps() works."""
    casted_dict = {}
    for label_str, (lat_val, lon_val) in center_points.items():
        # Convert float32 -> float
        casted_dict[label_str] = (float(lat_val), float(lon_val))
    return json.dumps(casted_dict)


def load_individual_detection_files(
    year_input_dir, use_env_filter, y_dim_name, x_dim_name
):
    """
    Load a sequence of detection result NetCDF files.
    Optimized: Loads coordinates and builds the global grid_info template
    only once to save memory.

    Args:
        year_input_dir (str): Directory containing the detection files for a specific year.
        use_env_filter (bool): Flag indicating whether to load environmental variable regions.
        y_dim_name (str): Name of the 1D y-dimension (from config).
        x_dim_name (str): Name of the 1D x-dimension (from config).

    Returns:
        tuple: (detection_results_list, grid_info)
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

                # Initialize the grid_info template on the very first file
                if grid_info is None:
                    # 1. Load dimensions using exact config names (No more guessing)
                    lat_1d = ds[y_dim_name].values
                    lon_1d = ds[x_dim_name].values

                    # 2. Extract 2D Geographic Coordinates (Always present in our new detection files)
                    lat2d = ds["latitude"].values
                    lon2d = ds["longitude"].values

                    # 3. Build the comprehensive grid template (including area_map)
                    grid_info = build_grid_info(
                        ds, y_dim_name, x_dim_name, lat2d, lon2d
                    )

                final_labeled_regions = ds["final_labeled_regions"].values[0]

                # Reconstruct Center Points
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
                        import json

                        center_points_dict = json.loads(ds.attrs["center_points_t0"])
                        if isinstance(center_points_dict, str):
                            center_points_dict = json.loads(center_points_dict)
                    except:
                        center_points_dict = {}

                # Create lightweight dictionary WITHOUT redundant coords
                detection_result = {
                    "final_labeled_regions": final_labeled_regions,
                    "time": time_val,
                    "center_points": center_points_dict,
                }

                if use_env_filter:
                    if "lifted_index_regions" in ds:
                        detection_result["lifted_index_regions"] = ds[
                            "lifted_index_regions"
                        ].values[0]
                    else:
                        detection_result["lifted_index_regions"] = np.zeros_like(
                            final_labeled_regions
                        )

                detection_results.append(detection_result)

        except Exception as e:
            logger.error(f"Error loading {filepath}: {e}")
            continue

    detection_results.sort(key=lambda x: x["time"])
    return detection_results, grid_info


def save_detection_result(detection_result, output_dir, data_source, grid_info):
    """
    Saves a single timestep's detection results to a compressed, CF-compliant NetCDF file.

    This function uses the grid_info template to dynamically apply 100% CF-compliant
    projection metadata while preserving the 2D detection masks and scalar center points.

    The output files are organized into a directory structure: `{output_dir}/YYYY/MM/`.

    Args:
        detection_result (dict): A dictionary containing all detection data for one frame:
            - "time" (datetime-like): The timestamp for this frame.
            - "final_labeled_regions" (np.ndarray): 2D array of detected convective objects.
            - "lifted_index_regions" (np.ndarray): 2D array of the environmental mask.
            - "center_points" (dict): A mapping `{label_id: (lat, lon)}` for all detected objects.
        output_dir (str): The root directory where output subfolders will be created.
        data_source (str): A string describing the input source.
        grid_info (dict): The verified spatial grid template.

    Returns:
        None
    """
    time_raw = detection_result["time"]
    try:
        # Works for standard times (ERA5, IMERG)
        time_obj = pd.to_datetime(time_raw).round("s")
    except (TypeError, ValueError):
        # Fallback for cftime objects (CORDEX, CMIP)
        time_obj = time_raw.item() if hasattr(time_raw, "item") else time_raw

    year_str = time_obj.strftime("%Y")
    month_str = time_obj.strftime("%m")

    structured_dir = os.path.join(output_dir, year_str, month_str)
    os.makedirs(structured_dir, exist_ok=True)

    filename = f"detection_{time_obj.strftime('%Y%m%dT%H%M')}.nc"
    output_filepath = os.path.join(structured_dir, filename)

    # 1. Extract dimensions from grid_info (The new way)
    y_dim = grid_info["y_dim_name"]
    x_dim = grid_info["x_dim_name"]
    y_1d = grid_info["lat1d"]
    x_1d = grid_info["lon1d"]

    # 2. Process Center Points (From your old function)
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

    # 3. Extract Grids with time dimension
    final_labeled_regions = np.expand_dims(
        detection_result["final_labeled_regions"], axis=0
    )
    lifted_index_regions = np.expand_dims(
        detection_result["lifted_index_regions"], axis=0
    )

    # 4. Create Dataset structure
    data_vars = {
        "final_labeled_regions": (["time", y_dim, x_dim], final_labeled_regions),
        "lifted_index_regions": (["time", y_dim, x_dim], lifted_index_regions),
        "label_id": (["labels"], label_ids),
        "label_lat": (["labels"], label_lats),
        "label_lon": (["labels"], label_lons),
    }

    ds = xr.Dataset(
        data_vars=data_vars,
        coords={
            "time": [time_obj],
            y_dim: y_1d,
            x_dim: x_1d,
        },
    )

    # Always add true 2D geographic coordinates from grid_info
    ds["latitude"] = ((y_dim, x_dim), grid_info["lat2d"])
    ds["longitude"] = ((y_dim, x_dim), grid_info["lon2d"])

    # 5. Metadata and Passthrough
    ds.attrs = {
        "title": "EMMA-Tracker Detection Output",
        "institution": "Wegener Center for Climate and Global Change, University of Graz",
        "source": data_source,
        "history": f"Created on {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "references": "Kneidinger et al. (2025)",
        "Conventions": "CF-1.7",
        "project": "EMMA",
    }

    ds["time"].attrs = {"standard_name": "time"}
    ds["latitude"].attrs = {"standard_name": "latitude", "units": "degrees_north"}
    ds["longitude"].attrs = {"standard_name": "longitude", "units": "degrees_east"}

    # ---------------------------------------------------------
    # STRICT NCVIEW SANITIZATION
    # ---------------------------------------------------------
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

        # Hard-force ONLY the exact 3 attributes ncview supports
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

    for var in ["final_labeled_regions", "lifted_index_regions"]:
        if var in ds:
            ds[var].attrs["grid_mapping"] = var_name
            ds[var].attrs["coordinates"] = "latitude longitude"
            ds[var].attrs["cell_methods"] = "time: point"

    ds["final_labeled_regions"].attrs.update(
        {"long_name": "Labeled Convective Regions", "units": "1"}
    )
    ds["lifted_index_regions"].attrs.update(
        {"long_name": "Environmental Mask", "units": "1"}
    )
    ds["label_id"].attrs.update({"long_name": "Feature Label IDs"})

    save_dataset_to_netcdf(ds, output_filepath)


def save_tracking_result(
    tracking_data_for_timestep, output_dir, data_source, grid_info, config=None
):
    """Saves a single timestep's tracking results to a compressed, CF-compliant NetCDF file.

    This function dynamically adapts to any grid projection (Regular, Rotated, Lambert, etc.)
    using the provided `grid_info` template, ensuring full GIS compatibility.
    It saves both the 2D segmentation masks (gridded data) and the scalar summary
    statistics of active tracks (tabular data) in a single file.

    The output files are organized into a directory structure: `{output_dir}/YYYY/MM/`.

    Args:
        tracking_data_for_timestep (dict): A dictionary containing all tracking data for one frame:
            - "time" (datetime-like): The timestamp for this frame.
            - "robust_mcs_id" (np.ndarray): 2D array of Track IDs for mature/robust MCS phases.
            - "mcs_id" (np.ndarray): 2D array of Track IDs for the full MCS lifecycle.
            - "mcs_id_merge_split" (np.ndarray): 2D array of Track IDs including merger history.
            - "tracking_centers" (dict): A mapping `{track_id: (lat, lon)}` for all active tracks.
        output_dir (str): The root directory where output subfolders will be created.
        data_source (str): A string describing the input source (e.g., "IMERG + ERA5" or "CERRA").
        grid_info (dict): The verified spatial grid template (generated via grid_manager.py), containing
            1D dimensions, 2D coordinates, and CF-compliant projection metadata.
        config (dict, optional): The run configuration dictionary. If provided, it is
            serialized into the global attribute 'run_configuration' for provenance.

    Output NetCDF Structure:
        Dimensions:
            time: 1 (unlimited)
            <y_dim_name>: Number of y-axis indices (dynamically named from config)
            <x_dim_name>: Number of x-axis indices (dynamically named from config)
            tracks: Number of active tracks in this specific timestep

        Variables:
            robust_mcs_id (time, y, x): Gridded track IDs (compressed).
            mcs_id (time, y, x): Gridded track IDs (compressed).
            mcs_id_merge_split (time, y, x): Gridded track IDs (compressed).
            active_track_id (tracks): List of IDs present in this file.
            active_track_lat (tracks): Center latitude for each ID.
            active_track_lon (tracks): Center longitude for each ID.
            active_track_touches_boundary (tracks): Flag (0/1) if system touches domain edge.

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

    # 1. Extract dimensions from grid_info (The new way)
    y_dim = grid_info["y_dim_name"]
    x_dim = grid_info["x_dim_name"]
    y_1d = grid_info["lat1d"]
    x_1d = grid_info["lon1d"]

    # 2. Process Center Points and Boundary Flags (From your old function)
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

    # 3. Extract Grids with time dimension
    robust_mcs_id_arr = np.expand_dims(
        tracking_data_for_timestep["robust_mcs_id"], axis=0
    )
    mcs_id_arr = np.expand_dims(tracking_data_for_timestep["mcs_id"], axis=0)
    mcs_id_merge_split_arr = np.expand_dims(
        tracking_data_for_timestep["mcs_id_merge_split"], axis=0
    )

    # 4. Create Dataset structure
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

    # Always add true 2D geographic coordinates from grid_info
    ds["latitude"] = ((y_dim, x_dim), grid_info["lat2d"])
    ds["longitude"] = ((y_dim, x_dim), grid_info["lon2d"])

    # 5. Metadata and Passthrough
    ds.attrs = {
        "title": "EMMA-Tracker Output",
        "institution": "Wegener Center for Climate and Global Change, University of Graz",
        "source": data_source,
        "history": f"Created on {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "Conventions": "CF-1.7",
        "project": "EMMA",
    }

    if config:
        try:
            ds.attrs["run_configuration"] = json.dumps(config, default=str)
        except Exception as e:
            logger.warning(f"Failed to serialize config: {e}")

    ds["time"].attrs = {"standard_name": "time"}
    ds["latitude"].attrs = {"standard_name": "latitude", "units": "degrees_north"}
    ds["longitude"].attrs = {"standard_name": "longitude", "units": "degrees_east"}

    # ---------------------------------------------------------
    # STRICT NCVIEW SANITIZATION
    # ---------------------------------------------------------
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

        # Hard-force ONLY the exact 3 attributes ncview supports
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