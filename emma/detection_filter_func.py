import numpy as np
import operator

OP_MAP = {
    ">=": operator.ge,
    "<=": operator.le,
    ">": operator.gt,
    "<": operator.lt,
}


def environmental_filter(
    env_var_array,
    labeled_regions,
    percentage,
    env_var_threshold=-2,
    operator_str="<=",
):
    """
    Checks each cluster in labeled_regions and returns a binary array where each grid point
    is marked 1 if its cluster has at least 'percentage' fraction of points meeting the environmental variable criteria.

    Parameters:
      env_var_array (numpy.ndarray): 2D float array containing the environmental variable with the same shape as labeled_regions.
      labeled_regions (numpy.ndarray): 2D integer array of cluster labels (0 is background).
      percentage (float): Fraction of grid points in a cluster that must satisfy the threshold.
      env_var_threshold (float): Threshold value for the environmental variable.
      operator_str (str): Threshold comparison operator ('>=', '<=', '>', '<').

    Returns:
      final_labeled_regions (numpy.ndarray): 2D binary array (same shape as labeled_regions) with 1 on grid points where
                                              the corresponding cluster passes the environmental criterion, 0 elsewhere.
    """
    op = OP_MAP.get(operator_str)
    if op is None:
        raise ValueError(
            f"Invalid operator '{operator_str}'. Must be one of {sorted(list(OP_MAP.keys()))}."
        )

    final_labeled_regions = np.zeros_like(labeled_regions)
    unique_labels = list(np.unique(labeled_regions))
    if 0 in unique_labels:
        unique_labels.remove(0)

    for unique_label in unique_labels:
        cluster_mask = labeled_regions == unique_label
        cluster_vars = env_var_array[cluster_mask]
        total_points = cluster_vars.size
        convective_points = np.sum(op(cluster_vars, env_var_threshold))
        if total_points > 0 and convective_points / total_points >= percentage:
            final_labeled_regions[cluster_mask] = 1
    return final_labeled_regions


def filter_mcs_candidates(clusters, core_plumes, min_size_threshold, min_nr_plumes):
    """
    Filter clusters to identify MCS candidates based on area and number of core plumes.

    Parameters:
    - clusters: 2D array of cluster labels.
    - core_plumes: 2D array of core plume labels.
    - min_size_threshold: Minimum size threshold for clusters (number of grid cells).
    - min_nr_plumes: Minimum number of core plumes required for MCS candidate.

    Returns:
    - mcs_candidate_labels: List of cluster labels that meet the MCS criteria.
    """
    mcs_candidate_labels = []
    cluster_labels = np.unique(clusters)
    cluster_labels = cluster_labels[cluster_labels != 0]

    for label_value in cluster_labels:
        cluster_mask = clusters == label_value
        plumes_in_cluster = np.unique(core_plumes[cluster_mask])
        num_plumes = len(plumes_in_cluster[plumes_in_cluster != 0])

        if np.sum(cluster_mask) >= min_size_threshold and num_plumes >= min_nr_plumes:
            mcs_candidate_labels.append(label_value)

    return mcs_candidate_labels


def compute_cluster_centers_of_mass(final_labeled_regions, lat, lon, main_var):
    """Computes the main_var-weighted center of mass for each labeled cluster.

    Args:
        final_labeled_regions (numpy.ndarray):
            2D integer array of cluster labels (excluding -1 or 0 for background).
        lat (numpy.ndarray):
            2D array of latitudes, same shape as final_labeled_regions.
        lon (numpy.ndarray):
            2D array of longitudes, same shape as final_labeled_regions.
        main_var (numpy.ndarray):
            2D array of main variable values, same shape as final_labeled_regions.

    Returns:
        dict:
            A dictionary mapping 'label_value' -> (center_lat, center_lon) in degrees.
            If a cluster has zero total main_var intensity, we store (np.nan, np.nan).
    """
    cluster_centers = {}
    unique_labels = np.unique(final_labeled_regions)
    # Exclude background label if it's 0 or -1
    unique_labels = unique_labels[unique_labels > 0]
    unique_labels = unique_labels.astype(str)  # Ensure keys are strings for JSON

    for label_value in unique_labels:
        mask = final_labeled_regions == int(label_value)
        # main_var for this cluster only
        cluster_vals = np.where(mask, main_var, 0.0)
        total_val = np.sum(cluster_vals)

        if total_val > 0:
            lat_weighted = np.sum(lat * cluster_vals)
            lon_weighted = np.sum(lon * cluster_vals)
            center_lat = lat_weighted / total_val
            center_lon = lon_weighted / total_val
            cluster_centers[label_value] = (center_lat, center_lon)
        else:
            # If zero total main_var, we cannot define a weighted center
            cluster_centers[label_value] = (np.nan, np.nan)

    return cluster_centers
