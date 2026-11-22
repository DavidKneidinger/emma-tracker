"""Multi-Sensor Convective Tracking Merger Module.

This module provides the topological fusion engine for EMMA-Tracker,
combining precipitation-based convective core tracks with satellite infrared
cloud shield tracks into unified mesoscale convective systems.

Physical Principles
-------------------
1. Independent Evolution:
   Convective precipitation features and cold-cloud canopies evolve independently
   until spatial co-location occurs.
2. 80% Containment & Forward Absorption (Option A):
   A precipitation track is merged with a co-located cloud shield if at least
   80% of the precipitation feature's instantaneous area is engulfed by the shield:
       Area(P ∩ Tb) / Area(P) >= 0.80
   Merging is evaluated strictly between P and Tb tracks. At merger, the system
   with the larger instantaneous active area absorbs the smaller system from
   that timestep forward without retroactively altering history.
3. Physical Component Delineation:
   The merged system footprint is the spatial union (Tb ∪ Precip).
   The system lifecycle is classified into:
     - 0: Background
     - 1: Initiation Precursor (Canopy active before surface rain)
     - 2: Active Precipitation (Precipitation feature area)
     - 3: Active Cloud Canopy (Cloud shield outside active rain features)
     - 4: Dissipation Anvil (Debris canopy remaining after rain ends)
4. Strict Filtering:
   Non-precipitating infrared cloud shields (pure Tb tracks) are discarded.
"""

from collections import defaultdict
from datetime import datetime, timedelta
import logging
from pathlib import Path
from typing import Any, Dict, Generator, List, Set, Tuple
import numpy as np
import yaml

from emma.input_output_merger import build_tracking_filepath, read_raw_tracking_slice

logger = logging.getLogger(__name__)


def get_mask_bbox(mask: np.ndarray) -> Tuple[int, int, int, int]:
  """Calculates the bounding box of a 2D boolean mask (rmin, rmax, cmin, cmax)."""
  rows = np.any(mask, axis=1)
  cols = np.any(mask, axis=0)
  rmin, rmax = np.where(rows)[0][[0, -1]]
  cmin, cmax = np.where(cols)[0][[0, -1]]
  return int(rmin), int(rmax), int(cmin), int(cmax)


def check_p_tb_overlap(
    m_p: np.ndarray,
    b_p: Tuple[int, int, int, int],
    area_p: int,
    m_t: np.ndarray,
    b_t: Tuple[int, int, int, int],
    min_overlap_ratio: float = 0.80,
) -> bool:
  """Tests whether a Tb cloud shield covers at least `min_overlap_ratio` (80%)

  of the precipitation feature area (Option A: Area(P ∩ Tb) / Area(P) >= 0.80).
  """
  if area_p <= 0:
    return False

  # Fast bounding box rejection: bounding boxes must strictly overlap
  if (
      b_p[1] < b_t[0]
      or b_p[0] > b_t[1]
      or b_p[3] < b_t[2]
      or b_p[2] > b_t[3]
  ):
    return False

  # Crop strictly to the intersection window
  r0 = max(b_p[0], b_t[0])
  r1 = min(b_p[1], b_t[1]) + 1
  c0 = max(b_p[2], b_t[2])
  c1 = min(b_p[3], b_t[3]) + 1

  sub_p = m_p[r0:r1, c0:c1]
  sub_t = m_t[r0:r1, c0:c1]

  overlap_pixels = np.count_nonzero(sub_p & sub_t)
  return (overlap_pixels / area_p) >= min_overlap_ratio


class MultiSensorMerger:
  """Topological merging engine for multi-sensor convective tracking in EMMA."""

  def __init__(self, config_path: Path):
    """Initializes the merger using config_merge.yaml."""
    with open(config_path, "r") as f:
      self.cfg = yaml.safe_load(f)

    self.dt_hours = float(self.cfg.get("dt_hours", 0.5))
    self.months = self.cfg.get("months", list(range(5, 12)))

    self.precip_dir = Path(self.cfg["precip_tracking_dir"])
    self.tb_dir = Path(self.cfg["tb_tracking_dir"])
    self.precip_var_name = self.cfg.get("precip_var_name", "mcs_id")
    self.tb_var_name = self.cfg.get("tb_var_name", "mcs_id")
    self.discard_pure_tb = self.cfg.get("discard_pure_tb", True)
    self.min_overlap_ratio = float(self.cfg.get("min_overlap_ratio", 0.80))

    self.time_step = timedelta(hours=self.dt_hours)

  def build_yearly_timesteps(self, year: int) -> List[datetime]:
    """Generates all configured timesteps for a specific calendar year."""
    timesteps = []
    for month in sorted(self.months):
      if month == 12:
        next_m_dt = datetime(year + 1, 1, 1, 0, 0)
      else:
        next_m_dt = datetime(year, month + 1, 1, 0, 0)

      curr = datetime(year, month, 1, 0, 0)
      while curr < next_m_dt:
        timesteps.append(curr)
        curr += self.time_step

    return timesteps

  def _resolve_topology_pass(
      self, timesteps: List[datetime]
  ) -> Tuple[
      List[datetime],
      Dict[datetime, Dict[Tuple[str, int], int]],
      Dict[int, Set[datetime]],
      Set[int],
      Dict[str, Any],
  ]:
    """Pass 1: Chronological scan resolving forward topological absorption."""
    logger.info("Pass 1: Scanning %d timesteps for mergers...", len(timesteps))

    valid_timesteps = []
    parent_sys = {}

    def find_sys(s: int) -> int:
      if parent_sys.get(s, s) == s:
        return s
      parent_sys[s] = find_sys(parent_sys[s])
      return parent_sys[s]

    track_assigned_sys = {}
    track_to_sys_at_t = defaultdict(dict)
    sys_rain_dts = defaultdict(set)
    sys_member_tracks = defaultdict(set)
    grid_meta = {}

    for idx, dt in enumerate(timesteps):
      p_file = build_tracking_filepath(self.precip_dir, dt)
      t_file = build_tracking_filepath(self.tb_dir, dt)

      if not (p_file.exists() and t_file.exists()):
        continue

      valid_timesteps.append(dt)

      raw_p, raw_t, _, meta = read_raw_tracking_slice(
          p_file, t_file, self.precip_var_name, self.tb_var_name
      )
      if "lat" not in grid_meta:
        grid_meta = meta

      active_p_ids = np.unique(raw_p[raw_p > 0])
      active_t_ids = np.unique(raw_t[raw_t > 0])

      # Vectorized area counting
      p_counts = (
          np.bincount(raw_p.ravel())
          if len(active_p_ids) > 0
          else np.array([0])
      )
      t_counts = (
          np.bincount(raw_t.ravel())
          if len(active_t_ids) > 0
          else np.array([0])
      )

      active_tracks = []
      track_masks = {}
      track_bboxes = {}
      track_areas = {}

      for pid in active_p_ids:
        k = ("P", int(pid))
        m = raw_p == pid
        active_tracks.append(k)
        track_masks[k] = m
        track_bboxes[k] = get_mask_bbox(m)
        track_areas[k] = int(p_counts[pid])

      for tid in active_t_ids:
        k = ("T", int(tid))
        m = raw_t == tid
        active_tracks.append(k)
        track_masks[k] = m
        track_bboxes[k] = get_mask_bbox(m)
        track_areas[k] = int(t_counts[tid])

      for k in active_tracks:
        if k not in track_assigned_sys:
          track_assigned_sys[k] = k[1]
          parent_sys[k[1]] = k[1]

      # Separate tracks by sensor
      active_p_tracks = [k for k in active_tracks if k[0] == "P"]
      active_t_tracks = [k for k in active_tracks if k[0] == "T"]
      adj_tracks = defaultdict(set)

      # Evaluate 80% containment strictly between P and Tb
      for kp in active_p_tracks:
        mp = track_masks[kp]
        bp = track_bboxes[kp]
        ap = track_areas[kp]

        for kt in active_t_tracks:
          mt = track_masks[kt]
          bt = track_bboxes[kt]

          if check_p_tb_overlap(
              mp, bp, ap, mt, bt, min_overlap_ratio=self.min_overlap_ratio
          ):
            adj_tracks[kp].add(kt)
            adj_tracks[kt].add(kp)

      # Graph resolution
      visited = set()
      for trk in active_tracks:
        if trk not in visited:
          comp = []
          stack = [trk]
          visited.add(trk)
          while stack:
            curr_trk = stack.pop()
            comp.append(curr_trk)
            for neighbor in adj_tracks[curr_trk]:
              if neighbor not in visited:
                visited.add(neighbor)
                stack.append(neighbor)

          touching_systems = list(
              set(find_sys(track_assigned_sys[t_node]) for t_node in comp)
          )

          # Dominant system absorbs smaller systems by instantaneous active area
          if len(touching_systems) > 1:
            sys_current_area = {}
            for s in touching_systems:
              sys_current_area[s] = sum(
                  track_areas[t_node]
                  for t_node in active_tracks
                  if find_sys(track_assigned_sys[t_node]) == s
              )

            winner_sys = max(
                touching_systems, key=lambda s: sys_current_area[s]
            )
            for s in touching_systems:
              if s != winner_sys:
                parent_sys[s] = winner_sys

      for k in active_tracks:
        effective_sys = find_sys(track_assigned_sys[k])
        track_to_sys_at_t[dt][k] = effective_sys
        sys_member_tracks[effective_sys].add(k)
        if k[0] == "P":
          sys_rain_dts[effective_sys].add(dt)

      if (idx + 1) % 1000 == 0 or (idx + 1) == len(timesteps):
        logger.info(
            "Pass 1 Progress: %d/%d timesteps scanned.",
            idx + 1,
            len(timesteps),
        )

    all_resolved_systems = set()
    for dt in valid_timesteps:
      all_resolved_systems.update(track_to_sys_at_t[dt].values())

    valid_systems = set()
    for s in all_resolved_systems:
      has_precip = any(k[0] == "P" for k in sys_member_tracks[s])
      if has_precip or not self.discard_pure_tb:
        valid_systems.add(s)

    logger.info(
        "Pass 1 Complete: %d raw systems identified, %d active systems"
        " retained.",
        len(all_resolved_systems),
        len(valid_systems),
    )

    return (
        valid_timesteps,
        track_to_sys_at_t,
        sys_rain_dts,
        valid_systems,
        grid_meta,
    )

  def stream_merged_year(
      self, year: int
  ) -> Generator[Dict[str, Any], None, None]:
    """Streams unified 2D fields one timestep at a time for Pass 2."""
    timesteps = self.build_yearly_timesteps(year)
    if not timesteps:
      logger.warning(
          "No timesteps found for year %d with months %s", year, self.months
      )
      return

    (
        valid_timesteps,
        track_to_sys_at_t,
        sys_rain_dts,
        valid_systems,
        grid_meta,
    ) = self._resolve_topology_pass(timesteps)

    if not valid_timesteps:
      logger.warning("No tracking files found for year %d.", year)
      return

    logger.info(
        "Pass 2: Streaming merged tracking fields for %d timesteps...",
        len(valid_timesteps),
    )

    n_lats = len(grid_meta["lat"])
    n_lons = len(grid_meta["lon"])

    for dt in valid_timesteps:
      p_file = build_tracking_filepath(self.precip_dir, dt)
      t_file = build_tracking_filepath(self.tb_dir, dt)

      raw_p, raw_t, time_val, _ = read_raw_tracking_slice(
          p_file, t_file, self.precip_var_name, self.tb_var_name
      )

      merged_system_id = np.zeros((n_lats, n_lons), dtype=np.int32)
      component_mask = np.zeros((n_lats, n_lons), dtype=np.int8)

      active_p_ids = np.unique(raw_p[raw_p > 0])
      active_t_ids = np.unique(raw_t[raw_t > 0])

      systems_at_this_dt = set()

      for tid in active_t_ids:
        k = ("T", int(tid))
        sys_id = track_to_sys_at_t[dt].get(k)
        if sys_id in valid_systems:
          merged_system_id[raw_t == tid] = sys_id
          systems_at_this_dt.add(sys_id)

      for pid in active_p_ids:
        k = ("P", int(pid))
        sys_id = track_to_sys_at_t[dt].get(k)
        if sys_id in valid_systems:
          merged_system_id[raw_p == pid] = sys_id
          systems_at_this_dt.add(sys_id)

      # Assign physical component states (0..4)
      for sys_id in systems_at_this_dt:
        sys_mask = merged_system_id == sys_id
        rain_dts = sys_rain_dts[sys_id]

        if not rain_dts:
          component_mask[sys_mask] = 3
          continue

        t_rain_start = min(rain_dts)
        t_rain_end = max(rain_dts)

        if dt < t_rain_start:
          component_mask[sys_mask] = 1  # Initiation Precursor
        elif dt > t_rain_end:
          component_mask[sys_mask] = 4  # Dissipation Anvil
        else:
          component_mask[sys_mask] = 3  # Active Cloud Canopy
          for pid in active_p_ids:
            k = ("P", int(pid))
            if track_to_sys_at_t[dt].get(k) == sys_id:
              component_mask[raw_p == pid] = 2  # Active Precipitation Feature

      yield {
          "datetime": dt,
          "time_val": time_val,
          "merged_system_id": merged_system_id,
          "system_component_mask": component_mask,
          "precip_track_id": raw_p,
          "tb_track_id": raw_t,
          "grid_meta": grid_meta,
      }