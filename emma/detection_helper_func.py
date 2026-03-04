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


def morphological_expansion_with_merging(
    core_labels, precip, expand_threshold=0.1, max_iterations=400
):
    """
    Performs iterative morphological expansion of labeled cores. In each iteration:
      1) Dilate each label by one pixel (8-connected).
      2) Collect newly added pixels (where precip >= expand_threshold).
      3) Build collisions (pixels claimed by multiple labels).
      4) Repeatedly merge collisions until no further merges occur.
      5) Finally apply expansions to the core_labels.
    Repeats until no more pixels are added or until max_iterations is reached.

    Args:
        core_labels (np.ndarray): 2D integer array of core labels (>0) and background=0
        precip (np.ndarray): 2D precipitation array (same shape as core_labels)
        expand_threshold (float): Minimum precipitation required to add new pixels
        max_iterations (int): Maximum expansion iterations

    Returns:
        np.ndarray: Updated label array after expansions and merges

    Notes:
        - Collisions are unified by default if precip >= expand_threshold at the collision pixel.
        - The merges are done in sets (transitive merges).
        - We re-check collisions repeatedly within each iteration to avoid partial merges
          (which can cause 'checkerboard' leftovers).
    """
    logger = logging.getLogger(__name__)
    structure = generate_binary_structure(2, 1)  # 8-connected
    iteration = 0

    while iteration < max_iterations:
        iteration += 1
        changed_pixels_total = 0

        # 1) Gather expansions for each label
        expansions = {lbl: set() for lbl in np.unique(core_labels) if lbl > 0}

        # Morphologically dilate each label by 1 step
        for lbl in expansions:
            feature_mask = core_labels == lbl
            dilated_mask = binary_dilation(feature_mask, structure=structure)
            # Only accept new pixels where precip >= expand_threshold
            new_pixels = dilated_mask & (~feature_mask) & (precip >= expand_threshold)

            if np.any(new_pixels):
                coords = zip(*new_pixels.nonzero())
                for r, c in coords:
                    expansions[lbl].add((r, c))

            changed_pixels_total += np.count_nonzero(new_pixels)

        if changed_pixels_total == 0:
            if iteration > max_iterations:
                logger.warning("Expansion stopped after max iteration: %d", iteration)
            break

        # 2) Merge collisions in a loop until stable
        merges_happened = True
        while merges_happened:
            merges_happened = False

            # 2a) Build a mapping from pixel -> list of labels claiming it
            pixel_claims = defaultdict(list)
            for lbl, pixset in expansions.items():
                for r, c in pixset:
                    # The expanding label claims it
                    pixel_claims[(r, c)].append(lbl)
                    
                    # THE FIX: Check if another label already owns this pixel
                    existing_lbl = core_labels[r, c]
                    if existing_lbl > 0 and existing_lbl != lbl:
                        if existing_lbl not in pixel_claims[(r, c)]:
                            pixel_claims[(r, c)].append(existing_lbl)

            # 2b) Detect collisions
            merges = []
            for (r, c), claim_list in pixel_claims.items():
                if len(claim_list) > 1 and precip[r, c] >= expand_threshold:
                    merges.append(set(claim_list))

            if not merges:
                break  # no collisions => done merging

            merges_to_apply = unify_merge_sets(merges)

            # 2c) Apply merges => pick smallest label as master
            for mg in merges_to_apply:
                if len(mg) < 2:
                    continue  # single label
                master = min(mg)
                old_labels = [x for x in mg if x != master]

                # Reassign expansions
                for old_lbl in old_labels:
                    if old_lbl in expansions:
                        expansions[master].update(expansions[old_lbl])
                        del expansions[old_lbl]
                        merges_happened = True

                # Also rewrite label array so we don't keep partial expansions
                # (this ensures collisions are recognized properly if they happen again)
                for old_lbl in old_labels:
                    core_labels[core_labels == old_lbl] = master

        # 3) Finally, apply expansions to core_labels
        #    Now that merges are stable for this iteration
        for lbl, pixset in expansions.items():
            for r, c in pixset:
                core_labels[r, c] = lbl

    return core_labels


def unify_merge_sets(merges):
    """
     Merges overlapping sets of labels transitively.
     For example, if merges = [ {1,2}, {2,3}, {4,5}, {1,3} ],
     the end result is [ {1,2,3}, {4,5} ].

    Args:
        merges (list of set): Each set contains labels that must unify.

     Returns:
         list of set: The final merged sets after transitive unification.

      Args:
          merges (list): list of sets, e.g. [ {1,2}, {2,3}, ... ]
    """
    merged = []
    for mset in merges:
        # compare with existing sets in 'merged'
        found = False
        for i, existing in enumerate(merged):
            if mset & existing:
                merged[i] = existing.union(mset)
                found = True
                break
        if not found:
            merged.append(mset)
    # repeat until stable
    stable = False
    while not stable:
        stable = True
        new_merged = []
        for s in merged:
            merged_any = False
            for i, x in enumerate(new_merged):
                if s & x:
                    new_merged[i] = x.union(s)
                    merged_any = True
                    stable = False
                    break
            if not merged_any:
                new_merged.append(s)
        merged = new_merged
    return merged
