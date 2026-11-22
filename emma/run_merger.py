"""Standalone Multi-Sensor Merging Runner for EMMA-Tracker.

Loads `config_merge.yaml` and executes the yearly multi-sensor fusion pipeline,
streaming time slices from `MultiSensorMerger` directly into `YearlyMergedNCWriter`.
"""

import argparse
from functools import partial
import logging
import multiprocessing as mp
from pathlib import Path
from typing import Tuple
import yaml

from emma.input_output_merger import YearlyMergedNCWriter, detect_common_years
from emma.multisensor_merger import MultiSensorMerger

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("emma.merge")


def process_single_year(
    year: int, config_path: Path
) -> Tuple[int, bool, int]:
  """Worker function processing a single calendar year into a yearly NetCDF file.

  Parameters
  ----------
  year : int
      Calendar year to process.
  config_path : Path
      Path to config_merge.yaml.

  Returns
  -------
  Tuple[int, bool, int]
      (year, success_flag, total_timesteps_written)
  """
  with open(config_path, "r") as f:
    cfg = yaml.safe_load(f)

  output_dir = Path(cfg["output_dir"])
  output_nc = output_dir / f"EMMA_merged_tracking_{year}.nc"
  complevel = int(cfg.get("complevel", 4))
  dt_hours = float(cfg.get("dt_hours", 0.5))

  logger.info("[%d] Starting merge task -> %s", year, output_nc)

  merger = MultiSensorMerger(config_path)
  stream = merger.stream_merged_year(year)

  writer = None
  steps_written = 0

  try:
    for idx, frame in enumerate(stream):
      if writer is None:
        writer = YearlyMergedNCWriter(
            output_path=output_nc,
            grid_meta=frame["grid_meta"],
            year=year,
            dt_hours=dt_hours,
            complevel=complevel,
        )

      writer.append_slice(
          time_val=frame["time_val"],
          merged_system_id=frame["merged_system_id"],
          component_mask=frame["system_component_mask"],
          precip_track_id=frame["precip_track_id"],
          tb_track_id=frame["tb_track_id"],
      )
      steps_written += 1

      if (idx + 1) % 1000 == 0:
        logger.info("[%d] Appended %d timesteps.", year, idx + 1)

  finally:
    if writer is not None:
      writer.close()
      logger.info(
          "[%d] Successfully finished (%d steps).", year, steps_written
      )
      return year, True, steps_written
    else:
      logger.warning("[%d] No valid timesteps found.", year)
      return year, False, 0


def main():
  """CLI entry point for running multi-sensor merging."""
  parser = argparse.ArgumentParser(
      description="EMMA Multi-Sensor Tracking Merger (Parallel Runner)"
  )
  parser.add_argument(
      "--config",
      type=Path,
      default=Path("config_merge.yaml"),
      help="Path to dedicated merge config YAML (default: config_merge.yaml)",
  )
  parser.add_argument(
      "--year",
      type=int,
      default=None,
      help="Optional specific year override (e.g. --year 2024)",
  )
  parser.add_argument(
      "--cores",
      type=int,
      default=None,
      help="Number of parallel worker processes (overrides config if set)",
  )
  args = parser.parse_args()

  if not args.config.exists():
    raise FileNotFoundError(f"Configuration file not found: {args.config}")

  with open(args.config, "r") as f:
    cfg = yaml.safe_load(f)

  # 1. Resolve Target Years (Handle years: [] by auto-detecting common years)
  if args.year is not None:
    years_to_run = [args.year]
  else:
    configured_years = cfg.get("years", [])
    if configured_years:
      years_to_run = [int(y) for y in configured_years]
    else:
      precip_dir = Path(cfg["precip_tracking_dir"])
      tb_dir = Path(cfg["tb_tracking_dir"])
      logger.info(
          "years: [] specified. Auto-detecting available years from %s and"
          " %s...",
          precip_dir,
          tb_dir,
      )
      years_to_run = detect_common_years(precip_dir, tb_dir)
      if not years_to_run:
        logger.error(
            "No common tracking years found between precip and Tb directories."
        )
        return

  logger.info("Target years to process: %s", years_to_run)

  # 2. Determine Core Count
  max_workers = (
      args.cores
      if args.cores is not None
      else int(cfg.get("number_of_cores", 10))
  )
  n_workers = max(1, min(len(years_to_run), max_workers))

  logger.info("Launching pool across %d CPU cores...", n_workers)

  # 3. Execute Parallel Processing Across Years
  worker_fn = partial(process_single_year, config_path=args.config)

  if n_workers == 1:
    for y in years_to_run:
      worker_fn(y)
  else:
    with mp.Pool(processes=n_workers) as pool:
      results = pool.map(worker_fn, years_to_run)
      for y, success, count in results:
        status = "SUCCESS" if success else "SKIPPED/FAILED"
        logger.info("Year %d: %s (%d steps)", y, status, count)


if __name__ == "__main__":
  main()