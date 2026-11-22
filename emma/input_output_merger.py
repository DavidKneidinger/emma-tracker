"""Input/Output Management for EMMA Multi-Sensor Merging.

This module isolates all reading and writing tasks for multi-sensor convective
fusion in EMMA. It provides utilities to locate raw files, auto-detect common
available years, and encapsulate CF-1.7 compliant yearly NetCDF-4 streaming.
"""

from datetime import datetime
import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple
import netCDF4 as nc
import numpy as np

logger = logging.getLogger(__name__)


def detect_common_years(precip_dir: Path, tb_dir: Path) -> List[int]:
  """Detects all calendar years present in both tracking directories.

  Parameters
  ----------
  precip_dir : Path
      Directory containing precipitation tracking year subfolders.
  tb_dir : Path
      Directory containing Tb tracking year subfolders.

  Returns
  -------
  List[int]
      Sorted list of integer years available in both directories.
  """
  precip_dir = Path(precip_dir)
  tb_dir = Path(tb_dir)

  if not precip_dir.exists() or not tb_dir.exists():
    logger.warning("One or both tracking directories do not exist.")
    return []

  p_years = {
      int(p.name)
      for p in precip_dir.iterdir()
      if p.is_dir() and p.name.isdigit() and len(p.name) == 4
  }
  t_years = {
      int(p.name)
      for p in tb_dir.iterdir()
      if p.is_dir() and p.name.isdigit() and len(p.name) == 4
  }

  common_years = sorted(list(p_years & t_years))
  return common_years


def build_tracking_filepath(base_dir: Path, dt: datetime) -> Path:
  """Constructs the standard EMMA tracking file path for a given datetime.

  Parameters
  ----------
  base_dir : Path
      Root directory of the tracking dataset.
  dt : datetime
      Timestamp of the tracking step.

  Returns
  -------
  Path
      Full path formatted as: base_dir/YYYY/MM/tracking_YYYYMMDDTHHMM.nc
  """
  y_str = dt.strftime("%Y")
  m_str = dt.strftime("%m")
  dt_str = dt.strftime("%Y%m%dT%H%M")
  return base_dir / y_str / m_str / f"tracking_{dt_str}.nc"


def read_raw_tracking_slice(
    p_file: Path,
    t_file: Path,
    p_var_name: str = "mcs_id",
    t_var_name: str = "mcs_id",
) -> Tuple[np.ndarray, np.ndarray, float, Dict[str, Any]]:
  """Reads 2D tracking masks and metadata from raw Precip and Tb NetCDF files.

  Parameters
  ----------
  p_file : Path
      Path to raw precipitation tracking file.
  t_file : Path
      Path to raw Tb cloud canopy tracking file.
  p_var_name : str, default="mcs_id"
      Variable name for precipitation track IDs.
  t_var_name : str, default="mcs_id"
      Variable name for Tb track IDs.

  Returns
  -------
  raw_p : np.ndarray
      2D int32 array of precipitation track IDs.
  raw_t : np.ndarray
      2D int32 array of Tb track IDs.
  time_val : float
      Numeric timestamp coordinate value from the file.
  grid_meta : Dict[str, Any]
      Dictionary containing 1D coordinates ('lat', 'lon') and time attributes.
  """
  with nc.Dataset(p_file, "r") as ds_p, nc.Dataset(t_file, "r") as ds_t:
    grid_meta = {
        "lat": ds_p.variables["lat"][:],
        "lon": ds_p.variables["lon"][:],
        "time_units": ds_p.variables["time"].units,
        "time_calendar": getattr(
            ds_p.variables["time"], "calendar", "proleptic_gregorian"
        ),
    }

    raw_p = ds_p.variables[p_var_name][0, :, :].astype(np.int32).filled(-1)
    raw_t = ds_t.variables[t_var_name][0, :, :].astype(np.int32).filled(-1)
    time_val = float(ds_p.variables["time"][0])

  # Alignment fallback if Tb grid has an extra latitude row (e.g., 301 vs 300)
  if raw_t.shape[0] == 301 and raw_p.shape[0] == 300:
    raw_t = raw_t[1:, :]

  return raw_p, raw_t, time_val, grid_meta


class YearlyMergedNCWriter:
  """Stream writer for consolidated CF-1.7 yearly merged tracking NetCDF-4 files."""

  def __init__(
      self,
      output_path: Path,
      grid_meta: Dict[str, Any],
      year: int,
      dt_hours: float = 0.5,
      complevel: int = 4,
      source_desc: str = "IMERG precipitation and MERGIR Tb tracking",
  ):
    """Initializes the CF-1.7 NetCDF file and allocates time-slice chunked variables."""
    self.output_path = Path(output_path)
    self.output_path.parent.mkdir(parents=True, exist_ok=True)
    self.time_idx = 0

    self.ds = nc.Dataset(self.output_path, "w", format="NETCDF4")

    n_lats = len(grid_meta["lat"])
    n_lons = len(grid_meta["lon"])

    # 1. Dimensions
    self.ds.createDimension("time", None)  # Unlimited for appending
    self.ds.createDimension("lat", n_lats)
    self.ds.createDimension("lon", n_lons)

    # 2. 1D Coordinates
    v_time = self.ds.createVariable("time", "f8", ("time",))
    v_time.standard_name = "time"
    v_time.units = grid_meta.get(
        "time_units", "days since 1950-01-01 00:00:00"
    )
    v_time.calendar = grid_meta.get("time_calendar", "proleptic_gregorian")
    v_time.axis = "T"

    v_lat = self.ds.createVariable("lat", "f4", ("lat",))
    v_lat[:] = grid_meta["lat"]
    v_lat.standard_name = "latitude"
    v_lat.units = "degrees_north"
    v_lat.axis = "Y"

    v_lon = self.ds.createVariable("lon", "f4", ("lon",))
    v_lon[:] = grid_meta["lon"]
    v_lon.standard_name = "longitude"
    v_lon.units = "degrees_east"
    v_lon.axis = "X"

    # 2D Grid Coordinates (CF Compliant)
    lon_2d, lat_2d = np.meshgrid(grid_meta["lon"], grid_meta["lat"])
    v_lat2d = self.ds.createVariable(
        "latitude", "f4", ("lat", "lon"), zlib=True
    )
    v_lat2d[:, :] = lat_2d
    v_lat2d.standard_name = "latitude"
    v_lat2d.units = "degrees_north"

    v_lon2d = self.ds.createVariable(
        "longitude", "f4", ("lat", "lon"), zlib=True
    )
    v_lon2d[:, :] = lon_2d
    v_lon2d.standard_name = "longitude"
    v_lon2d.units = "degrees_east"

    v_crs = self.ds.createVariable("crs", "i4")
    v_crs.grid_mapping_name = "latitude_longitude"
    v_crs.semi_major_axis = 6378137.0
    v_crs.inverse_flattening = 298.257223563

    # Time-slice chunking: (1, n_lats, n_lons)
    chunks = (1, n_lats, n_lons)

    # 3. Merged Data Variables
    self.v_merged = self.ds.createVariable(
        "merged_system_id",
        "i4",
        ("time", "lat", "lon"),
        zlib=True,
        complevel=complevel,
        chunksizes=chunks,
        fill_value=-1,
    )
    self.v_merged.long_name = "Unified Merged Convective System Identifier"
    self.v_merged.units = "1"
    self.v_merged.grid_mapping = "crs"
    self.v_merged.coordinates = "latitude longitude"
    self.v_merged.cell_methods = "time: point"

    self.v_mask = self.ds.createVariable(
        "system_component_mask",
        "i1",
        ("time", "lat", "lon"),
        zlib=True,
        complevel=complevel,
        chunksizes=chunks,
        fill_value=0,
    )
    self.v_mask.long_name = "Physical System Lifecycle Component Mask"
    self.v_mask.units = "1"
    self.v_mask.grid_mapping = "crs"
    self.v_mask.coordinates = "latitude longitude"
    self.v_mask.cell_methods = "time: point"
    self.v_mask.flag_values = np.array([0, 1, 2, 3, 4], dtype=np.int8)
    self.v_mask.flag_meanings = (
        "background initiation_precursor active_rain_core "
        "active_cloud_canopy dissipation_anvil"
    )

    self.v_precip = self.ds.createVariable(
        "precip_track_id",
        "i4",
        ("time", "lat", "lon"),
        zlib=True,
        complevel=complevel,
        chunksizes=chunks,
        fill_value=-1,
    )
    self.v_precip.long_name = "Original IMERG Precipitation Track Identifier"
    self.v_precip.units = "1"
    self.v_precip.grid_mapping = "crs"
    self.v_precip.coordinates = "latitude longitude"
    self.v_precip.cell_methods = "time: point"

    self.v_tb = self.ds.createVariable(
        "tb_track_id",
        "i4",
        ("time", "lat", "lon"),
        zlib=True,
        complevel=complevel,
        chunksizes=chunks,
        fill_value=-1,
    )
    self.v_tb.long_name = "Original MERGIR Tb Cloud Shield Track Identifier"
    self.v_tb.units = "1"
    self.v_tb.grid_mapping = "crs"
    self.v_tb.coordinates = "latitude longitude"
    self.v_tb.cell_methods = "time: point"

    # 4. Global CF Attributes
    self.ds.title = "EMMA Unified Multi-Sensor Convective Tracking"
    self.ds.institution = (
        "Wegener Center for Climate and Global Change, University of Graz"
    )
    self.ds.source = source_desc
    self.ds.history = (
        f"Created on {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')} "
        "via EMMA Topological MultiSensorMerger"
    )
    self.ds.Conventions = "CF-1.7"
    self.ds.project = "EMMA"
    self.ds.dt_hours = float(dt_hours)
    self.ds.target_year = int(year)
    self.ds.containment_rule = (
        "Forward topological absorption by active area; "
        "non-precipitating cloud canopies discarded."
    )

  def append_slice(
      self,
      time_val: float,
      merged_system_id: np.ndarray,
      component_mask: np.ndarray,
      precip_track_id: np.ndarray,
      tb_track_id: np.ndarray,
  ):
    """Appends a 2D time slice to the NetCDF file."""
    self.ds.variables["time"][self.time_idx] = time_val
    self.v_merged[self.time_idx, :, :] = merged_system_id
    self.v_mask[self.time_idx, :, :] = component_mask
    self.v_precip[self.time_idx, :, :] = precip_track_id
    self.v_tb[self.time_idx, :, :] = tb_track_id
    self.time_idx += 1

  def close(self):
    """Flushes buffers and closes the NetCDF file."""
    if self.ds is not None and self.ds.isopen():
      self.ds.close()
      logger.info(
          "Finalized %s (%d timesteps).", self.output_path, self.time_idx
      )

  def __enter__(self):
    return self

  def __exit__(self, exc_type, exc_val, exc_tb):
    self.close()