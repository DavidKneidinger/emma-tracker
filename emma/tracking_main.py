#!/usr/bin/env python
"""
tracking_main.py

Main routine for tracking Mesoscale Convective Systems (MCSs) across multiple timesteps.
Tracks are assigned via spatial overlap, and a robust filtering based on an environmental variable
detection (provided in the detection results as 'env_var_regions') is applied.

The script returns per-timestep tracking arrays, main track IDs, lifetime arrays,
merging and splitting events, and tracking center positions.
"""

import numpy as np
import logging
from collections import defaultdict
from .tracking_filter_func import filter_main_mcs, filter_relevant_systems
from .tracking_helper_func import (
    assign_new_id,
    check_overlaps,
    handle_continuation,
    handle_no_overlap,
    compute_max_consecutive,
    attempt_advection_rescue,
)
from .tracking_merging import handle_merging
from .tracking_splitting import handle_splitting

logger = logging.getLogger(__name__)


def track_mcs(
    detection_results,
    grid_info,
    main_area_thresh,
    nmaxmerge,
    use_env_filter=True,
    dt_hours=1.0,
    main_lifetime_thresh_hours=4,
    **kwargs,
):
    """
    Tracks Mesoscale Convective Systems (MCSs) and filters them based on a combined set of criteria.

    This function first tracks all detected main variable features over time using spatial overlap,
    handling complex merging and splitting events. After the initial tracking, it performs a
    rigorous filtering step to identify "main MCSs". A track qualifies as a main MCS only if it
    contains a continuous period of at least 'main_lifetime_thresh_hours' hours where, simultaneously,
    its area is greater than 'main_area_thresh' and it meets the environmental criterion (if 'use_env_filter' is True).

    The function returns three distinct sets of track IDs representing different levels of filtering,
    from the most restrictive ("in-phase" MCSs) to the most inclusive ("full family tree").

    Args:
        detection_results (List[dict]): A list where each dictionary represents one timestep and contains:
            - "final_labeled_regions" (np.ndarray): 2D array of detected cluster labels.
            - "env_var_regions" (np.ndarray): 2D binary array where 1 indicates a cluster met environmental criteria.
            - "center_points" (dict): Mapping of cluster label to its (lat, lon) center. Optional.
            - "time" (datetime.datetime): Timestamp for the data.
            - "lat2d" (np.ndarray): 2D array of latitudes.
            - "lon2d" (np.ndarray): 2D array of longitudes.
            - "lat" (np.ndarray): 1D array of latitudes.
            - "lon" (np.ndarray): 1D array of longitudes.
        grid_info (dict): dictionary containing the globally verified spatial dimensions and area map.
        main_area_thresh (float): The minimum area (in km²) a track must have to be considered in its mature phase.
        nmaxmerge (int): The maximum number of parent systems to consider in a single merging event.
        use_env_filter (bool): If True, enables the environmental conditioning check.
        dt_hours (float, optional): Time step resolution in hours (e.g., 0.5 for 30 minutes, 1.0 for 1 hour). Defaults to 1.0.
        main_lifetime_thresh_hours (int): The minimum number of consecutive hours a track must simultaneously meet criteria.

    Returns:
        Tuple: A tuple containing the organized tracking results.
    """
    # Support backward compatibility for legacy keyword args
    if "use_li_filter" in kwargs:
        use_env_filter = kwargs["use_li_filter"]

    previous_labeled_regions = None
    previous_cluster_ids = {}
    merge_split_cluster_ids = {}
    next_cluster_id = 1

    mcs_ids_list = []
    lifetime_list = []
    tracking_centers_list = []
    time_list = []

    lat = grid_info["lat1d"]
    lon = grid_info["lon1d"]
    lat2d = grid_info["lat2d"]
    lon2d = grid_info["lon2d"]
    grid_area_map_km2 = grid_info["area_map"]

    lifetime_dict = defaultdict(int)
    max_area_dict = defaultdict(float)

    merging_events = []
    splitting_events = []

    # Dictionary to track robust flag for each assigned track ID.
    robust_flag_dict = {}
    convective_history = defaultdict(dict)

    # Compute min number of frames required for a track based on lifetime threshold and resolution.
    min_frames = max(1, int(round(main_lifetime_thresh_hours / dt_hours)))

    # Determine if environmental filtering is available
    use_env = use_env_filter and "env_var_regions" in detection_results[0]

    for idx, detection_result in enumerate(detection_results):
        final_labeled_regions = detection_result["final_labeled_regions"]
        center_points_dict = detection_result.get("center_points", {})
        current_time = detection_result["time"]

        # Get environmental regions if available.
        if use_env:
            env_regions = detection_result.get(
                "env_var_regions", detection_result.get("env_var_regions")
            )
        else:
            env_regions = None

        # Initialize ID and lifetime arrays for current timestep.
        mcs_id = np.zeros_like(final_labeled_regions, dtype=np.int32)
        mcs_lifetime = np.zeros_like(final_labeled_regions, dtype=np.int32)

        unique_labels = np.unique(final_labeled_regions)
        unique_labels = unique_labels[unique_labels != 0]

        if len(unique_labels) == 0:
            logger.info(f"No clusters detected at {current_time}")
            previous_cluster_ids = {}
            previous_labeled_regions = None

            mcs_ids_list.append(mcs_id)
            lifetime_list.append(mcs_lifetime)
            time_list.append(current_time)
            tracking_centers_list.append({})
            continue

        if previous_labeled_regions is None:
            # First timestep with clusters: assign new track IDs.
            for label in unique_labels:
                cluster_mask = final_labeled_regions == label
                area = np.sum(grid_area_map_km2[cluster_mask])
                assigned_id, next_cluster_id = assign_new_id(
                    label,
                    cluster_mask,
                    area,
                    next_cluster_id,
                    lifetime_dict,
                    max_area_dict,
                    mcs_id,
                    mcs_lifetime,
                )
                previous_cluster_ids[label] = assigned_id
                merge_split_cluster_ids[label] = assigned_id
                if use_env:
                    meets_env = np.all(env_regions[cluster_mask] == 1)
                else:
                    meets_env = True
                robust_flag_dict[assigned_id] = meets_env
                convective_history[assigned_id][idx] = meets_env
        else:
            # Subsequent timesteps: check overlaps between previous and current clusters.
            overlap_map = check_overlaps(
                previous_labeled_regions,
                final_labeled_regions,
                previous_cluster_ids,
                overlap_threshold=10,
            )
            temp_assigned = {}
            labels_no_overlap = [
                lbl for lbl, old_ids in overlap_map.items() if not old_ids
            ]

            if labels_no_overlap:
                # We need a map of only the real overlaps to calculate the flow vector
                clean_overlap_map = {
                    lbl: ids for lbl, ids in overlap_map.items() if ids
                }
                rescued_overlaps = attempt_advection_rescue(
                    labels_no_overlap,
                    previous_labeled_regions,
                    final_labeled_regions,
                    previous_cluster_ids,
                    clean_overlap_map,  # Pass the clean map
                    grid_area_map_km2,
                    overlap_threshold=10,
                )

                if rescued_overlaps:
                    # Add the rescued overlaps back to the main map for processing
                    overlap_map.update(rescued_overlaps)
                    rescued_labels = set(rescued_overlaps.keys())
                    labels_no_overlap = [
                        lbl for lbl in labels_no_overlap if lbl not in rescued_labels
                    ]

            for new_lbl, old_ids in overlap_map.items():
                if len(old_ids) == 0:
                    if new_lbl not in labels_no_overlap:
                        labels_no_overlap.append(new_lbl)
                elif len(old_ids) == 1:
                    chosen_id = old_ids[0]
                    handle_continuation(
                        new_label=new_lbl,
                        old_track_id=chosen_id,
                        final_labeled_regions=final_labeled_regions,
                        mcs_id=mcs_id,
                        mcs_lifetime=mcs_lifetime,
                        lifetime_dict=lifetime_dict,
                        max_area_dict=max_area_dict,
                        grid_area_map_km2=grid_area_map_km2,
                    )
                    temp_assigned[new_lbl] = chosen_id
                    if use_env:
                        current_mask = final_labeled_regions == new_lbl
                        current_env_pass = np.all(env_regions[current_mask] == 1)
                    else:
                        current_env_pass = True
                    robust_flag_dict[chosen_id] = (
                        robust_flag_dict.get(chosen_id, False) or current_env_pass
                    )
                    convective_history[chosen_id][idx] = current_env_pass
                else:
                    # Merging: handle multiple overlapping previous clusters.
                    chosen_id = handle_merging(
                        new_label=new_lbl,
                        old_track_ids=old_ids,
                        merging_events=merging_events,
                        final_labeled_regions=final_labeled_regions,
                        current_time=current_time,
                        max_area_dict=max_area_dict,
                        grid_area_map_km2=grid_area_map_km2,
                        nmaxmerge=nmaxmerge,
                    )
                    mask = final_labeled_regions == new_lbl
                    mcs_id[mask] = chosen_id
                    lifetime_dict[chosen_id] += 1
                    temp_assigned[new_lbl] = chosen_id
                    if use_env:
                        current_mask = final_labeled_regions == new_lbl
                        current_env_pass = np.all(env_regions[current_mask] == 1)
                    else:
                        current_env_pass = True
                    robust_flag = (
                        any(robust_flag_dict.get(old_id, False) for old_id in old_ids)
                        or current_env_pass
                    )
                    robust_flag_dict[chosen_id] = robust_flag
                    convective_history[chosen_id][idx] = current_env_pass

            # Handle clusters with no overlap.
            new_assign_map, next_cluster_id = handle_no_overlap(
                labels_no_overlap,
                final_labeled_regions,
                next_cluster_id,
                lifetime_dict,
                max_area_dict,
                mcs_id,
                mcs_lifetime,
                grid_area_map_km2,
            )
            temp_assigned.update(new_assign_map)

            # Handle splitting events.
            oldid_to_newlist = defaultdict(list)
            for lbl, tid in temp_assigned.items():
                oldid_to_newlist[tid].append(lbl)
            for old_id, newlbls in oldid_to_newlist.items():
                if len(newlbls) > 1:
                    splitted_map, next_cluster_id = handle_splitting(
                        old_id,
                        newlbls,
                        final_labeled_regions,
                        current_time,
                        next_cluster_id,
                        splitting_events,
                        mcs_id,
                        mcs_lifetime,
                        lifetime_dict,
                        max_area_dict,
                        grid_area_map_km2,
                        nmaxsplit=nmaxmerge,
                    )
                    for nl, finalid in splitted_map.items():
                        temp_assigned[nl] = finalid
                        current_mask = final_labeled_regions == nl
                        if use_env:
                            is_env_pass = np.all(env_regions[current_mask] == 1)
                            robust_flag_dict[finalid] = is_env_pass
                        else:
                            robust_flag_dict[finalid] = True
                        logger.info(
                            f"Track splitting at {current_time} for parent track {old_id}. "
                            f"New child track {finalid} assigned robust flag: {robust_flag_dict[finalid]}"
                        )
            current_cluster_ids = temp_assigned
            previous_cluster_ids = current_cluster_ids
            logger.info(f"MCS tracking at {current_time} processed.")

        previous_labeled_regions = final_labeled_regions.copy()

        mcs_ids_list.append(mcs_id)
        lifetime_list.append(mcs_lifetime)
        time_list.append(current_time)

        # Build tracking centers for this timestep.
        centers_this_timestep = {}
        label_by_cluster = defaultdict(list)
        for lbl, tid in previous_cluster_ids.items():
            label_by_cluster[tid].append(lbl)
        for tid, label_list in label_by_cluster.items():
            center_latlon = (None, None)
            for detect_label in label_list:
                detect_label_str = str(int(detect_label))
                if detect_label_str in center_points_dict:
                    center_latlon = center_points_dict[detect_label_str]
                    break
            centers_this_timestep[str(tid)] = center_latlon
        tracking_centers_list.append(centers_this_timestep)

    # ---- Final Filtering Step ----
    logger.info("Starting efficient final filtering of tracks...")

    # Pass 1: Pre-compute properties for all tracks at each timestep.
    track_properties_by_time = defaultdict(dict)
    for i, mcs_id_array in enumerate(mcs_ids_list):
        unique_ids_in_frame = np.unique(mcs_id_array)
        unique_ids_in_frame = unique_ids_in_frame[unique_ids_in_frame > 0]

        for tid in unique_ids_in_frame:
            mask = mcs_id_array == tid
            area = np.sum(grid_area_map_km2[mask])

            meets_env = convective_history[tid].get(i, False) if use_env else True

            track_properties_by_time[tid][i] = {
                "meets_area": area >= main_area_thresh,
                "meets_env": meets_env,
            }

    # Pass 2: Use pre-computed data to quickly build boolean series and find main MCSs.
    mcs_ids = []
    for tid in list(lifetime_dict.keys()):

        bool_series_area = []
        bool_series_env = []

        for i in range(len(mcs_ids_list)):
            props = track_properties_by_time.get(tid, {}).get(i)
            if props:
                bool_series_area.append(props["meets_area"])
                bool_series_env.append(props["meets_env"])
            else:
                bool_series_area.append(False)
                bool_series_env.append(False)

        # Condition 1: Check if mature phase (based on area) meets lifetime threshold.
        if compute_max_consecutive(bool_series_area) >= min_frames:

            env_during_mature_phase_list = [
                area and env for area, env in zip(bool_series_area, bool_series_env)
            ]

            # Condition 2: Environmental condition met at least once during mature phase.
            if any(env_during_mature_phase_list):
                mcs_ids.append(tid)

    logger.info(
        f"Tracking identified {len(mcs_ids)} main MCSs after combined filtering."
    )

    # --- Generate the 3 Final Output Variables ---
    # Output 3 (Most Inclusive): Full "Family Tree"
    mcs_id_merge_split = filter_relevant_systems(
        mcs_ids_list, mcs_ids, merging_events, splitting_events
    )

    # Output 2 (Intermediate): Full Lifetime of Main MCSs
    mcs_id = filter_main_mcs(mcs_ids_list, mcs_ids)

    # Output 1 (Most Restrictive): "In-Phase" MCSs
    robust_mcs_id = []
    for i, mcs_id_array in enumerate(mcs_id):
        frame_in_phase = mcs_id_array.copy()
        unique_ids_in_frame = np.unique(frame_in_phase[frame_in_phase > 0])

        for tid in unique_ids_in_frame:
            props = track_properties_by_time.get(tid, {}).get(i)
            if not props or not props["meets_area"]:
                frame_in_phase[frame_in_phase == tid] = 0

        robust_mcs_id.append(frame_in_phase)

    return (
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
    )
