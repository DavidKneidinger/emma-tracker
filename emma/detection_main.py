import numpy as np
import logging
from .input_output import load_main_var_data, load_env_var_data
from .detection_helper_func import (
    smooth_field,
    detect_cores_connected,
    expand_cores,
)
from .detection_filter_func import (
    filter_mcs_candidates,
    environmental_filter,
    compute_cluster_centers_of_mass,
)


def detect_mcs_in_file(
    main_var_file_path,
    main_var_data_var,
    env_var_file_path,
    env_var_data_var,
    lat_name,
    lon_name,
    core_threshold,
    env_var_threshold,
    envelope_threshold,
    min_size_threshold,
    min_nr_plumes,
    env_var_percentage,
    grid_info,
    main_time_index=0,
    env_time_index=0,
):
    """
    Detect MCSs in a single file.

    Parameters:
    - main_var_file_path: Path to the NetCDF file containing the main variable data.
    - main_var_data_var: Variable name of the main variable.
    - env_var_file_path: Path to the NetCDF file containing the environmental variable data.
    - env_var_data_var: Variable name of the environmental variable.
    - lat_name: Name of the latitude coordinate variable.
    - lon_name: Name of the longitude coordinate variable.
    - core_threshold: Threshold for core detection.
    - env_var_threshold: Threshold for the environmental variable.
    - envelope_threshold: Threshold for the system envelope.
    - min_size_threshold: Minimum size threshold for clusters (number of grid cells).
    - min_nr_plumes: Minimum number of convective plumes required for MCS candidate.
    - env_var_percentage: Percentage of area that needs to fulfill the environmental variable criteria.
    - grid_info: Dictionary containing the globally verified spatial dimensions and area map.
    - main_time_index: Index of the time step to process in main variable file.
    - env_time_index: Index of the time step to process in environmental variable file.

    Returns:
    - detection_result: Dictionary containing detection results.
    """
    logger = logging.getLogger(__name__)
    env_var_regions = None

    # Load data
    ds, _, _, _, _, main_var = load_main_var_data(
        main_var_file_path, main_var_data_var, lat_name, lon_name, main_time_index
    )

    lat2d = grid_info["lat2d"]
    lon2d = grid_info["lon2d"]
    lat = grid_info["lat1d"]
    lon = grid_info["lon1d"]

    # Initialize env_var_regions as an array initialized with the threshold
    # This ensures it always has the correct shape and type for your output format.
    env_var_regions = np.ones_like(main_var, dtype=np.int32) * env_var_threshold

    # Step 1: Smooth the main variable field
    main_var_smooth = smooth_field(main_var)

    # Step 2: Detect core regions with connected component labeling
    core_labels = detect_cores_connected(
        main_var_smooth,
        core_thresh=core_threshold,
        min_cluster_size=4,  # Min number of points in a cluster
    )

    # Step 3: Group cores into continuous systems via masking
    expanded_labels = expand_cores(
        core_labels,
        main_var_smooth,
        expand_threshold=envelope_threshold,
    )

    # Step 4: Filter MCS candidates based on number of core plumes, size, and environmental filter
    mcs_candidate_labels = filter_mcs_candidates(
        expanded_labels, core_labels, min_size_threshold, min_nr_plumes
    )

    # Create final labeled regions for MCS candidates
    final_labeled_regions = np.where(
        np.isin(expanded_labels, mcs_candidate_labels), expanded_labels, 0
    )

    if (
        env_var_file_path and env_var_file_path.strip()
    ):  # Check if path is not None and not just empty spaces
        logger.info("Environmental variable file provided. Applying filter...")
        # Load the environmental variable data
        _, _, _, _, _, env_var_converted = load_env_var_data(
            env_var_file_path,
            env_var_data_var,
            lat_name,
            lon_name,
            env_time_index,
        )
        # Apply the filter
        env_var_regions = environmental_filter(
            env_var_converted.values
            if hasattr(env_var_converted, "values")
            else env_var_converted,
            final_labeled_regions,
            env_var_percentage,
            env_var_threshold=env_var_threshold,
        )
    else:
        env_var_regions = np.zeros_like(final_labeled_regions, dtype=np.uint8)
        logger.info("No environmental variable file provided. Skipping filter.")

    # Step 5: Compute cluster centers of mass
    cluster_centers = compute_cluster_centers_of_mass(
        final_labeled_regions, lat2d, lon2d, main_var
    )

    # Prepare detection result
    detection_result = {
        "final_labeled_regions": final_labeled_regions,
        "env_var_regions": env_var_regions,
        "lat2d": lat2d,
        "lon2d": lon2d,
        "lat": lat,
        "lon": lon,
        "main_var": main_var_smooth,
        "time": ds["time"].values,
        "convective_plumes": core_labels,
        "center_points": cluster_centers,
    }
    logger.info(f"MCS detection completed for {main_var_file_path}.")
    return detection_result
