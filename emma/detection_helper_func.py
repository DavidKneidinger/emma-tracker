import numpy as np
import logging
from collections import defaultdict
from scipy.ndimage import binary_dilation, generate_binary_structure, gaussian_filter
from skimage.measure import label as connected_label


def smooth_precipitation_field(
    precipitation: np.ndarray, sigma: float = 1.0
) -> np.ndarray:
    """
    Apply a Gaussian filter to smooth a 2D field.

    This method uses a Gaussian kernel for smoothing, which is generally
    preferred for scientific applications over a simple box filter as it
    provides isotropic (radially symmetric) smoothing and avoids introducing
    high-frequency artifacts.

    Parameters:
    - precipitation (np.ndarray): 2D array of precipitation values.
    - sigma (float): The standard deviation for the Gaussian kernel, given in
      units of grid cells. A larger sigma results in more smoothing.

    Returns:
    - np.ndarray: The smoothed precipitation field as a 2D array.
    """
    return gaussian_filter(precipitation, sigma=sigma, mode="reflect")


def detect_cores_connected(precipitation, core_thresh=10.0, min_cluster_size=3):
    """Cluster heavy precipitation cores using connected component labeling.

    This function thresholds the precipitation field at the specified core threshold
    and then identifies contiguous clusters using connected component analysis.
    Any connected component with fewer than `min_cluster_size` pixels is discarded.

    Args:
        precipitation (numpy.ndarray): 2D array representing the precipitation field.
        core_thresh (float, optional): Threshold for heavy precipitation cores (e.g., mm/h).
            Defaults to 10.0.
        min_cluster_size (int, optional): Minimum number of pixels required for a cluster to be kept.
            Clusters with fewer pixels than this threshold are discarded. Defaults to 3.

    Returns:
        numpy.ndarray: 2D array of integer cluster labels for each grid point.
            Pixels not belonging to any cluster are labeled as 0. Detected clusters are assigned
            consecutive positive integers starting at 1.
    """
    # Create a binary mask where precipitation meets or exceeds the core threshold.
    core_mask = precipitation >= core_thresh

    # If there are fewer pixels above threshold than the minimum cluster size, return an array of zeros.
    if np.sum(core_mask) < min_cluster_size:
        return np.zeros_like(precipitation, dtype=int)

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


def expand_cores(core_labels, precip, expand_threshold=1.0):
    """
    Groups convective cores into contiguous precipitation systems using a global mask.

    1) Creates a binary mask of all precipitation >= expand_threshold.
    2) Labels all 8-connected regions in this mask.
    3) Retains only those labeled regions that overlap with at least one heavy core.

    Args:
        core_labels (np.ndarray): 2D integer array of heavy cores (labels > 0, background = 0).
        precip (np.ndarray): 2D precipitation array.
        expand_threshold (float): Minimum precipitation defining the system envelope.

    Returns:
        np.ndarray: 2D integer array of the full storm systems.
                    Background is 0. Systems are labeled with consecutive integers.
    """
    logger = logging.getLogger(__name__)

    # 1. Create the global moderate precipitation mask.
    # Bitwise OR (|) ensures core pixels are explicitly included even if smoothed below threshold
    strat_mask = (precip >= expand_threshold) | (core_labels > 0)

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
