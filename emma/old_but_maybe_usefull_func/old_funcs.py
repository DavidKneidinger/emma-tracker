import numpy as np
from skimage.measure import regionprops
import hdbscan


def extract_shape_features(clusters, lat, lon, grid_spacing_km):
    """Extracts shape features (e.g., area, perimeter, axes) from labeled clusters.

    Args:
        clusters (numpy.ndarray):
            2D array of integer cluster labels (0 indicates background/no cluster).
        lat (numpy.ndarray):
            2D array of latitudes, same shape as `clusters`.
        lon (numpy.ndarray):
            2D array of longitudes, same shape as `clusters`.
        grid_spacing_km (float):
            Approximate grid spacing in kilometers for converting pixel-based measurements
            (like area in pixel count) into km².

    Returns:
        dict:
            A dictionary `shape_features` mapping each nonzero cluster label to a
            dictionary of shape properties. For example:

            shape_features[label_value] = {
                "area_km2": ...,
                "perimeter_km": ...,
                "major_axis_length_km": ...,
                "minor_axis_length_km": ...,
                "aspect_ratio": ...,
                "orientation_deg": ...,
                "solidity": ...,
                "eccentricity": ...,
                "extent": ...,
                "convex_area_km2": ...,
                "circularity": ...,
            }
    """
    shape_features = {}

    labeled_clusters = clusters.astype(int)
    cluster_labels = np.unique(labeled_clusters)
    cluster_labels = cluster_labels[cluster_labels != 0]  # Exclude background label = 0

    for label_value in cluster_labels:
        cluster_mask = labeled_clusters == label_value
        binary_image = cluster_mask.astype(int)

        # Compute region properties via skimage
        props_list = regionprops(binary_image)
        if len(props_list) == 0:
            continue
        props = props_list[0]  # Should be only one region per label_value

        # Convert region measurements to physical units
        area = props.area * (grid_spacing_km**2)  # km²
        perimeter = props.perimeter * grid_spacing_km  # km
        major_axis_length = props.major_axis_length * grid_spacing_km
        minor_axis_length = props.minor_axis_length * grid_spacing_km

        # Derived shape features
        aspect_ratio = (
            major_axis_length / minor_axis_length if minor_axis_length != 0 else np.nan
        )
        orientation_deg = np.degrees(props.orientation) % 360
        solidity = props.solidity
        eccentricity = props.eccentricity
        extent = props.extent
        convex_area = props.convex_area * (grid_spacing_km**2)
        if perimeter != 0:
            circularity = (4.0 * np.pi * area) / (perimeter**2)
        else:
            circularity = np.nan

        # Store base shape metrics
        shape_features[label_value] = {
            "area_km2": area,
            "perimeter_km": perimeter,
            "major_axis_length_km": major_axis_length,
            "minor_axis_length_km": minor_axis_length,
            "aspect_ratio": aspect_ratio,
            "orientation_deg": orientation_deg,
            "solidity": solidity,
            "eccentricity": eccentricity,
            "extent": extent,
            "convex_area_km2": convex_area,
            "circularity": circularity,
        }

    return shape_features


def classify_mcs_types(shape_features):
    """
    Classify MCS clusters into types based on shape features.

    Parameters:
    - shape_features: Dictionary of shape features per cluster.

    Returns:
    - mcs_classification: Dictionary with cluster labels as keys and MCS types as values.
    """
    mcs_classification = {}

    for label_value, features in shape_features.items():
        aspect_ratio = features["aspect_ratio"]
        area = features["area_km2"]
        circularity = features["circularity"]

        # Initialize type
        mcs_type = "Unclassified"

        # Classification rules
        if aspect_ratio >= 5 and features["major_axis_length_km"] >= 100:
            mcs_type = "Squall Line"
        elif aspect_ratio <= 2 and area >= 100000 and circularity >= 0.7:
            mcs_type = "MCC"
        elif 2 <= aspect_ratio < 5:
            mcs_type = "Linear MCS"
        else:
            mcs_type = "Other MCS"

        mcs_classification[label_value] = mcs_type

    return mcs_classification


def detect_cores_hdbscan(precipitation, lat, lon, core_thresh=10.0, min_cluster_size=3):
    """
    Cluster heavy precipitation cores using HDBSCAN.

    Parameters:
    - precipitation: 2D precipitation field.
    - lat: 2D array of latitude values corresponding to the precipitation grid.
    - lon: 2D array of longitude values corresponding to the precipitation grid.
    - core_thresh: Threshold for heavy precipitation cores.
    - min_cluster_size: Minimum number of samples in a cluster for HDBSCAN.

    Returns:
    - labeled_array: 2D array with cluster labels for each grid point.
      Points not belonging to any cluster are labeled as -1.
    """
    """
    Example, same as above but we pass lat2d, lon2d as arguments.
    """
    core_mask = precipitation >= core_thresh
    labels_2d = np.zeros_like(precipitation, dtype=int)
    if np.sum(core_mask) < min_cluster_size:
        return labels_2d

    # Extract lat/lon for the masked pixels
    core_coords = np.column_stack((lat[core_mask].ravel(), lon[core_mask].ravel()))

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        metric="haversine",
        allow_single_cluster=False,
    )
    clusterer.fit(np.radians(core_coords))
    # clusterer.labels_ = [-1, 0, 1, 2, ...]  (0 is the first cluster, -1 is noise)
    core_labels = np.where(
        clusterer.labels_ >= 0, clusterer.labels_ + 1, clusterer.labels_
    )
    core_labels[core_labels == -1] = 0

    # Insert into 2D array
    labels_2d[core_mask] = core_labels
    return labels_2d
