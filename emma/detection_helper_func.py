import numpy as np
import logging
from collections import defaultdict
from scipy.ndimage import gaussian_filter
from skimage.measure import label as connected_label
import operator

OP_MAP = {
    ">=": operator.ge,
    "<=": operator.le,
    ">": operator.gt,
    "<": operator.lt,
}


def smooth_field(main_var: np.ndarray, sigma: float = 1.0) -> np.ndarray:
    """
    Apply a Gaussian filter to smooth a 2D field.

    This method uses a Gaussian kernel for smoothing, which is generally
    preferred for scientific applications over a simple box filter as it
    provides isotropic (radially symmetric) smoothing and avoids introducing
    high-frequency artifacts.

    Parameters:
    - main_var (np.ndarray): 2D array of main variable values.
    - sigma (float): The standard deviation for the Gaussian kernel, given in
      units of grid cells. A larger sigma results in more smoothing.

    Returns:
    - np.ndarray: The smoothed main variable field as a 2D array.
    """
    return gaussian_filter(main_var, sigma=sigma, mode="reflect")


def detect_cores_connected(
    main_var, core_thresh=10.0, min_cluster_size=3, operator_str=">="
):
    """Cluster heavy cores using connected component labeling.

    This function thresholds the main variable field at the specified core threshold
    and then identifies contiguous clusters using connected component analysis.
    Any connected component with fewer than `min_cluster_size` pixels is discarded.

    Args:
        main_var (numpy.ndarray): 2D array representing the main variable field.
        core_thresh (float, optional): Threshold for core detection.
            Defaults to 10.0.
        min_cluster_size (int, optional): Minimum number of pixels required for a cluster to be kept.
            Clusters with fewer pixels than this threshold are discarded. Defaults to 3.
        operator_str (str): Threshold comparison operator ('>=' or '<=').

    Returns:
        numpy.ndarray: 2D array of integer cluster labels for each grid point.
            Pixels not belonging to any cluster are labeled as 0. Detected clusters are assigned
            consecutive positive integers starting at 1.
    """
    op = OP_MAP.get(operator_str)
    if op is None:
        raise ValueError(f"Invalid operator '{operator_str}'. Must be '>=' or '<='.")

    core_mask = op(main_var, core_thresh)

    # If there are fewer pixels above threshold than the minimum cluster size, return an array of zeros.
    if np.sum(core_mask) < min_cluster_size:
        return np.zeros_like(main_var, dtype=int)

    # Label connected components in the binary mask.
    # Use connectivity=2 for 8-connected neighborhood.
    labeled_components = connected_label(core_mask, connectivity=2)

    # Initialize final label array.
    final_labels = np.zeros_like(labeled_components, dtype=int)
    unique_labels = np.unique(labeled_components)
    # Exclude the background label (0)
    unique_labels = unique_labels[unique_labels != 0]

    # Reassign labels only for connected components that meet the min_cluster_size.
    current_label = 1
    for label_val in unique_labels:
        comp_mask = labeled_components == label_val
        if np.sum(comp_mask) >= min_cluster_size:
            final_labels[comp_mask] = current_label
            current_label += 1
        # Components smaller than min_cluster_size are discarded (remain 0).
    return final_labels


def expand_cores(core_labels, main_var, expand_threshold=1.0, operator_str=">="):
    """
    Groups convective cores into contiguous storm systems using a global mask.

    1) Creates a binary mask based on operator_str (e.g. main_var >= expand_threshold or <=).
    2) Labels all 8-connected regions in this mask.
    3) Retains only those labeled regions that overlap with at least one heavy core.

    Args:
        core_labels (np.ndarray): 2D integer array of heavy cores (labels > 0, background = 0).
        main_var (np.ndarray): 2D main variable array.
        expand_threshold (float): Minimum threshold defining the system envelope.
        operator_str (str): Threshold comparison operator ('>=' or '<=').

    Returns:
        np.ndarray: 2D integer array of the full storm systems.
                    Background is 0. Systems are labeled with consecutive integers.
    """
    logger = logging.getLogger(__name__)

    op = OP_MAP.get(operator_str)
    if op is None:
        raise ValueError(f"Invalid operator '{operator_str}'. Must be '>=' or '<='.")

    # 1. Create the global moderate main variable mask using the dynamic operator.
    # Bitwise OR (|) ensures core pixels are explicitly included even if smoothed below threshold
    strat_mask = op(main_var, expand_threshold) | (core_labels > 0)

    # 2. Label all 8-connected areas instantly using skimage (consistent with core detection)
    strat_labels = connected_label(strat_mask, connectivity=2)

    # 3. Find which stratiform labels overlap with our heavy cores
    core_mask = core_labels > 0
    valid_system_ids = np.unique(strat_labels[core_mask])

    # Remove the background (0)
    valid_system_ids = valid_system_ids[valid_system_ids > 0]

    if len(valid_system_ids) == 0:
        logger.warning("No expanding systems found matching the core criteria.")
        return np.zeros_like(core_labels)

    # 4. Filter the map to keep ONLY the valid systems
    final_labels = np.where(np.isin(strat_labels, valid_system_ids), strat_labels, 0)

    return final_labels
