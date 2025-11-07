import numpy as np
import datetime
from dataclasses import dataclass
from typing import List


@dataclass
class SplittingEvent:
    """Stores information about a splitting event in the tracking."""

    time: datetime.datetime
    parent_id: int
    child_ids: List[int]
    parent_area: float
    child_areas: List[float]


def handle_splitting(
    old_track_id,
    new_label_list,
    final_labeled_regions,
    current_time,
    next_cluster_id,
    splitting_events,
    mcs_id,
    mcs_lifetime,
    lifetime_dict,
    max_area_dict,
    grid_area_map_km2,
    nmaxsplit=5,
):
    """
    Resolves splitting when one old_track_id is claimed by multiple new labels.

    Args:
        old_track_id (int): The parent track ID from the previous timestep.
        new_label_list (List[int]): List of new detection labels that overlap old_track_id.
        final_labeled_regions (numpy.ndarray): Labeled regions from current timestep.
        current_time (datetime.datetime): Timestamp for splitting event.
        next_cluster_id (int): Next available track ID for ephemeral new IDs.
        splitting_events (List[SplittingEvent]): A list to record splitting events.
        mcs_id (numpy.ndarray): 2D array for track IDs.
        mcs_lifetime (numpy.ndarray): 2D array for track lifetimes.
        lifetime_dict (dict): Maps track ID -> number of timesteps.
        max_area_dict (dict): Maps track ID -> max area encountered.
        grid_area_map_km2 (np.ndarray): 2D array of grid cell areas (km²).
        nmaxsplit (int, optional): Max number of splits in a single event. Defaults to 5.

    Returns:
        Tuple[dict, int]: A dictionary { new_label : final assigned track ID } and the updated next_cluster_id.
    """
    if len(new_label_list) <= 1:
        return {}, next_cluster_id  # no actual split

    new_label_areas = []
    for nlbl in new_label_list:
        mask = final_labeled_regions == nlbl
        new_label_areas.append(np.sum(grid_area_map_km2[mask]))

    idx_sorted = sorted(
        range(len(new_label_list)), key=lambda i: new_label_areas[i], reverse=True
    )
    keep_idx = idx_sorted[0]
    keep_label = new_label_list[keep_idx]
    keep_area = new_label_areas[keep_idx]

    splitted_assign_map = {}

    # The largest child keeps old_track_id
    splitted_assign_map[keep_label] = old_track_id
    lifetime_dict[old_track_id] -= 1  # Patch for double counting
    if keep_area > max_area_dict[old_track_id]:
        max_area_dict[old_track_id] = keep_area

    keep_mask = final_labeled_regions == keep_label
    mcs_id[keep_mask] = old_track_id
    mcs_lifetime[keep_mask] = lifetime_dict[old_track_id]

    splitted_child_labels = []
    splitted_child_areas = []

    for i in idx_sorted[1:]:
        lbl = new_label_list[i]
        area_s = new_label_areas[i]
        splitted_assign_map[lbl] = next_cluster_id
        lifetime_dict[next_cluster_id] = 1
        max_area_dict[next_cluster_id] = area_s
        mask_s = final_labeled_regions == lbl
        mcs_id[mask_s] = next_cluster_id
        mcs_lifetime[mask_s] = 1
        splitted_child_labels.append(next_cluster_id)
        splitted_child_areas.append(area_s)
        next_cluster_id += 1

    if splitting_events is not None and len(new_label_list) > 1:
        if len(new_label_list) > nmaxsplit:
            new_label_list = new_label_list[:nmaxsplit]
        sevt = SplittingEvent(
            time=current_time,
            parent_id=old_track_id,
            child_ids=[splitted_assign_map[lbl] for lbl in new_label_list],
            parent_area=max_area_dict[old_track_id],
            child_areas=[
                max_area_dict[splitted_assign_map[lbl]] for lbl in new_label_list
            ],
        )
        splitting_events.append(sevt)

    return splitted_assign_map, next_cluster_id
