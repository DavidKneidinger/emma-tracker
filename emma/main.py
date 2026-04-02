# main.py

import os
import glob
import gc
import concurrent.futures
import argparse
import yaml
import sys
import logging
import pandas as pd
from collections import defaultdict

from .config import EmmaConfig
from .detection_main import detect_mcs_in_file
from .tracking_main import track_mcs
from .input_output import (
    setup_logging,
    handle_exception,
    build_task_list,
    save_detection_result,
    load_individual_detection_files,
    save_tracking_result,
)
from .grid_manager import verify_and_build_grid_template
from .postprocessing import run_postprocessing_year

# Set a global exception handler to log uncaught exceptions
sys.excepthook = handle_exception


def process_file(
    precip_file_path,
    precip_data_var,
    lifted_index_file_path,
    lifted_index_data_var,
    lat_name,
    lon_name,
    heavy_precip_threshold,
    lifted_index_threshold,
    moderate_precip_threshold,
    min_size_threshold,
    min_nr_plumes,
    lifted_index_percentage,
    grid_info,
    precip_time_index,
    li_time_index,
):
    """
    Wrapper function to run MCS detection for a single file.
    This is used as the target for parallel processing.
    """
    result = detect_mcs_in_file(
        precip_file_path,
        precip_data_var,
        lifted_index_file_path,
        lifted_index_data_var,
        lat_name,
        lon_name,
        heavy_precip_threshold,
        lifted_index_threshold,
        moderate_precip_threshold,
        min_size_threshold,
        min_nr_plumes,
        lifted_index_percentage,
        grid_info,
        precip_time_index=precip_time_index,
        li_time_index=li_time_index,
    )
    return result


def parse_arguments():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(description="MCS Detection and Tracking")
    parser.add_argument(
        "--config", type=str, required=True, help="Path to the configuration YAML file."
    )
    return parser.parse_args()


def main():
    """
    Main execution script for MCS detection and tracking.

    The workflow is structured as follows:
    1. Load configuration from a YAML file.
    2. Group input data files by year.
    3. Loop through each year for processing:
        a. Run MCS detection for every timestep, saving results to hourly files.
        b. Load the year's detection results back into memory.
        c. Run the tracking algorithm on the full year's data.
        d. Apply the postprocessing filter to the raw tracking data.
        d. Save the final tracking results to hourly files.
    This yearly batch approach ensures scalability for multi-year datasets.
    """
    # --- 1. SETUP AND CONFIGURATION ---
    logger = logging.getLogger(__name__)
    args = parse_arguments()

    # LOAD CONFIGURATION STRICTLY
    try:
        cfg = EmmaConfig.load(args.config)
    except Exception as e:
        logger.critical(f"Failed to load configuration: {e}")
        sys.exit(1)

    # General parameters (Access via cfg object)
    precip_data_dir = cfg.precip_data_directory
    detection_output_path = cfg.detection_output_path
    raw_tracking_output_dir = cfg.raw_tracking_output_dir
    tracking_output_dir = cfg.filtered_tracking_output_dir
    precip_template = cfg.precip_filename_template
    precip_data_var = cfg.precip_var_name
    lat_name = cfg.lat_name
    lon_name = cfg.lon_name
    data_source = cfg.data_source

    if cfg.detection_parameters.use_lifted_index:
        lifted_index_data_var = cfg.lifted_index_var_name
    else:
        lifted_index_data_var = False

    # Read optional date filtering parameters ---
    years_to_process = cfg.years
    months_to_process = cfg.months

    # Create output directories
    os.makedirs(detection_output_path, exist_ok=True)
    os.makedirs(raw_tracking_output_dir, exist_ok=True)

    # Track the file mode for each phase.
    log_modes = {"detection": "w", "tracking": "w", "postprocessing": "w"}

    # Initialize logging based on the starting phase
    if cfg.detection:
        setup_logging(
            detection_output_path, filename="detection.log", mode=log_modes["detection"]
        )
        log_modes["detection"] = "a"
    elif cfg.tracking:
        setup_logging(
            raw_tracking_output_dir, filename="tracking.log", mode=log_modes["tracking"]
        )
        log_modes["tracking"] = "a"
    elif cfg.postprocessing:
        os.makedirs(tracking_output_dir, exist_ok=True)
        setup_logging(
            tracking_output_dir,
            filename="postprocessing.log",
            mode=log_modes["postprocessing"],
        )
        log_modes["postprocessing"] = "a"

    # --- 2. FIND, FILTER, AND GROUP INPUT FILES ---
    tasks_by_year = defaultdict(list)

    if cfg.detection:
        logger.info("Building task list from directories and templates...")

        li_dir = (
            cfg.lifted_index_data_directory
            if cfg.detection_parameters.use_lifted_index
            else None
        )
        li_template = (
            cfg.lifted_index_filename_template
            if cfg.detection_parameters.use_lifted_index
            else None
        )

        all_tasks = build_task_list(
            precip_dir=precip_data_dir,
            precip_template=precip_template,
            li_dir=li_dir,
            li_template=li_template,
            years=years_to_process,
            months=months_to_process,
        )

        if not all_tasks:
            logger.warning("No valid time steps found matching the criteria. Exiting.")
            sys.exit(0)

        for task in all_tasks:
            tasks_by_year[task["aligned_time"].year].append(task)

    # Determine the years to iterate over
    if cfg.detection:
        years_to_iterate = sorted(tasks_by_year.keys())
    else:
        # If detection is skipped, determine years from config or existing output directories
        if years_to_process:
            years_to_iterate = sorted(years_to_process)
        else:
            years_to_iterate = []

            # Check detection output for existing year folders
            if os.path.exists(detection_output_path):
                subdirs = [
                    d
                    for d in os.listdir(detection_output_path)
                    if os.path.isdir(os.path.join(detection_output_path, d))
                ]
                for d in subdirs:
                    if d.isdigit():
                        years_to_iterate.append(int(d))

            # Check tracking output (useful if detection files were deleted to save space)
            if not years_to_iterate and os.path.exists(raw_tracking_output_dir):
                subdirs = [
                    d
                    for d in os.listdir(raw_tracking_output_dir)
                    if os.path.isdir(os.path.join(raw_tracking_output_dir, d))
                ]
                for d in subdirs:
                    if d.isdigit():
                        years_to_iterate.append(int(d))

            years_to_iterate = sorted(list(set(years_to_iterate)))

        if not years_to_iterate:
            logger.warning(
                "Detection is off and no years specified in config or found in output directories. Nothing to do."
            )

    # --- 3. MAIN YEARLY PROCESSING LOOP ---
    global_grid_template = None

    for year in years_to_iterate:
        logger.info(f"--- Starting processing for year: {year} ---")

        # --- 3a. DETECTION PHASE ---
        if cfg.detection:
            # Configure logging for detection
            setup_logging(
                detection_output_path,
                filename="detection.log",
                mode=log_modes["detection"],
            )
            log_modes["detection"] = "a"

            tasks_for_year = tasks_by_year.get(year, [])

            if not tasks_for_year:
                logger.warning(f"No detection tasks found for year {year}. Skipping.")
                continue

            # Build the global grid template using the very first task's files
            if global_grid_template is None:
                first_task = tasks_for_year[0]
                global_grid_template = verify_and_build_grid_template(
                    first_precip_file=first_task["precip_file"],
                    first_li_file=first_task.get("li_file"),
                    y_dim_name=lat_name,
                    x_dim_name=lon_name,
                )

            logger.info(
                f"Running detection for {len(tasks_for_year)} timesteps in {year}..."
            )

            if cfg.use_multiprocessing:
                with concurrent.futures.ProcessPoolExecutor(
                    max_workers=cfg.number_of_cores
                ) as executor:
                    futures = [
                        executor.submit(
                            process_file,
                            task["precip_file"],
                            precip_data_var,
                            task.get("li_file"),
                            lifted_index_data_var,
                            lat_name,
                            lon_name,
                            cfg.detection_parameters.heavy_precip_threshold,
                            cfg.detection_parameters.lifted_index_threshold,
                            cfg.detection_parameters.moderate_precip_threshold,
                            cfg.detection_parameters.min_size_threshold,
                            cfg.detection_parameters.min_nr_plumes,
                            cfg.detection_parameters.lifted_index_percentage_threshold,
                            global_grid_template,
                            task["precip_idx"],
                            task.get("li_idx"),
                        )
                        for task in tasks_for_year
                    ]
                    for future in concurrent.futures.as_completed(futures):
                        try:
                            detection_result = future.result()
                            save_detection_result(
                                detection_result,
                                detection_output_path,
                                data_source,
                                global_grid_template,
                            )
                        except Exception as e:
                            logger.error(f"A detection task failed: {e}")
            else:
                for task in tasks_for_year:
                    detection_result = detect_mcs_in_file(
                        task["precip_file"],
                        precip_data_var,
                        task.get("li_file"),
                        lifted_index_data_var,
                        lat_name,
                        lon_name,
                        cfg.detection_parameters.heavy_precip_threshold,
                        cfg.detection_parameters.lifted_index_threshold,
                        cfg.detection_parameters.moderate_precip_threshold,
                        cfg.detection_parameters.min_size_threshold,
                        cfg.detection_parameters.min_nr_plumes,
                        cfg.detection_parameters.lifted_index_percentage_threshold,
                        global_grid_template,
                        precip_time_index=task["precip_idx"],
                        li_time_index=task.get("li_idx"),
                    )
                    save_detection_result(
                        detection_result,
                        detection_output_path,
                        data_source,
                        global_grid_template,
                    )
            logger.info(f"Detection for year {year} finished.")
            print(f"Detection for year {year} finished.")

        # --- 3b. & 3c. TRACKING PHASE ---
        if cfg.tracking:
            # Configure logging for tracking
            setup_logging(
                raw_tracking_output_dir,
                filename="tracking.log",
                mode=log_modes["tracking"],
            )
            log_modes["tracking"] = "a"

            logger.info(f"Loading all detection files for year {year}...")

            year_detection_dir = os.path.join(detection_output_path, str(year))
            detection_results, tracking_grid_info = load_individual_detection_files(
                year_detection_dir,
                cfg.detection_parameters.use_lifted_index,
                cfg.lat_name,
                cfg.lon_name,
            )

            # Apply month filter if specified
            if months_to_process and detection_results:
                original_count = len(detection_results)

                def get_month_safely(t):
                    # Safely unpack 0-D numpy arrays if present
                    t_scalar = t.item() if hasattr(t, "item") else t
                    # cftime and python datetime objects have a .month attribute natively
                    if hasattr(t_scalar, "month"):
                        return t_scalar.month
                    # Fallback for standard numpy.datetime64 objects
                    return pd.to_datetime(t_scalar).month

                detection_results = [
                    res
                    for res in detection_results
                    if get_month_safely(res["time"]) in months_to_process
                ]
                logger.info(
                    f"Filtered detection results for specified months. Kept {len(detection_results)} of {original_count} files."
                )

            if not detection_results:
                logger.warning(
                    f"No detection results found for year {year} (or for the specified months). Skipping tracking."
                )
                continue

            # Tracking Phase
            logger.info(f"Starting tracking for year {year}...")
            (
                robust_mcs_id,
                mcs_id,
                mcs_id_merge_split,
                lifetime_list,
                time_list,
                lat2d,
                lon2d,
                lat,
                lon,
                merging_events,
                splitting_events,
                tracking_centers_list,
            ) = track_mcs(
                detection_results,
                tracking_grid_info,
                # Pass strict values from nested config
                cfg.tracking_parameters.main_lifetime_thresh,
                cfg.tracking_parameters.main_area_thresh,
                cfg.tracking_parameters.nmaxmerge,
                use_li_filter=cfg.detection_parameters.use_lifted_index,
            )

            # Saving Phase
            logger.info(f"Saving individual hourly tracking files for year {year}...")
            for i in range(len(time_list)):
                # Package all data for this single timestep into a dictionary
                tracking_data_for_timestep = {
                    "robust_mcs_id": robust_mcs_id[i],
                    "mcs_id": mcs_id[i],
                    "mcs_id_merge_split": mcs_id_merge_split[i],
                    "lifetime": lifetime_list[i],
                    "time": time_list[i],
                    "lat2d": lat2d,
                    "lon2d": lon2d,
                    "lat": lat,
                    "lon": lon,
                    "tracking_centers": tracking_centers_list[i],
                }
                # Pass full config object if save_tracking_result needs it, or just necessary parts
                # Assuming save_tracking_result expects 'config' (dict-like or object)
                save_tracking_result(
                    tracking_data_for_timestep,
                    raw_tracking_output_dir,
                    data_source,
                    tracking_grid_info,
                    cfg,
                )

            del detection_results
            del mcs_id
            del robust_mcs_id
            del lifetime_list
            gc.collect()
            logger.info(f"--- Finished tracking for year: {year} ---")
            print(f"--- Finished tracking for year: {year} ---")

        # --- 3e. POST-PROCESSING PHASE ---
        if not cfg.detection_parameters.use_lifted_index:
            print("Skipping postprocessing because of no lifted index...")
        else:
            lifted_index_data_var = cfg.lifted_index_var_name

            if cfg.postprocessing:

                # Configure logging for post-processing
                os.makedirs(tracking_output_dir, exist_ok=True)
                setup_logging(
                    tracking_output_dir,
                    filename="postprocessing.log",
                    mode=log_modes["postprocessing"],
                )
                log_modes["postprocessing"] = "a"

                logger.info("Logging initialized for POST-PROCESSING phase.")
                try:
                    run_postprocessing_year(
                        year,
                        raw_tracking_output_dir,
                        tracking_output_dir,
                        precip_data_var,
                        lifted_index_data_var,
                        lat_name,
                        lon_name,
                        cfg,
                    )
                except Exception as e:
                    logger.error(f"Post-processing failed for year {year}: {e}")

            logger.info(f"--- Finished processing for year: {year} ---")
            print(f"--- Finished processing for year: {year} ---")

    logger.info("All processing completed successfully.")
    print("All processing completed successfully.")


if __name__ == "__main__":
    main()
