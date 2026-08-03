# EMMA-Tracker: User Guide & Technical Documentation

This document provides detailed instructions on data preprocessing, configuration parameters, and the scientific logic behind the detection and tracking algorithm.

---

## 1. Data Preprocessing

The EMMA-Tracker uses a variable-agnostic architecture centered around two input roles:
1. **Main Tracking Variable (`main_var`)**: The primary feature detection field (e.g., Precipitation, Satellite $T_b$, or Radar Reflectivity).
2. **Environmental Variable (`env_var`)**: An optional secondary field used to condition or filter features based on thermodynamic stability or environment (e.g., Lifted Index, CAPE, CIN).

### Example Application: Precipitation & Atmospheric Instability

#### Primary Tracking Variable (e.g., Precipitation)
* **Format:** NetCDF (.nc).
* **Resolution:** The algorithm is tuned for high-resolution data (approx. 0.1° / 12km and finer).
* **Units:** e.g., mm/h (or kg m⁻² s⁻¹, the tool handles standard conversions).

#### Environmental Variable (e.g., Lifted Index - LI)
To use physics-based instability filtering, you can provide an environmental variable such as LI.
* **Source:** Typically derived from ERA5 or model level outputs.
* **Calculation:** Scripts to calculate LI from standard pressure levels (T, Q, Z) are located in the `../preprocess/` directory.
* **Grid:** Must match the main variable grid exactly.

**Note:** If your dataset lacks an environmental conditioning field or you are running satellite-only tracking, set `use_env_var: False` in `config.yaml`. The tracker will completely bypass environmental file loading and detect objects based solely on the main tracking field.

---

## 2. Configuration Guide (`config.yaml`)

The algorithm is controlled by a single YAML file. Below is a detailed explanation of the parameters.

### Data & Paths
| Parameter | Description |
| :--- | :--- |
| `main_var_data_directory` | Root folder containing main tracking variable NetCDF files. |
| `env_var_data_directory` | Root folder containing environmental variable NetCDF files. |
| `raw_tracking_output_dir` | Where initial tracking results (unfiltered) are saved. |
| `filtered_tracking_output_dir` | Where final, publication-ready MCS tracks are saved. |
| `main_var_name` | Name of the primary tracking variable in the NetCDF file (e.g., `"precipitation"` or `"tb"`). |
| `env_var_name` | Name of the environmental variable in the NetCDF file (e.g., `"LI"` or `"CAPE"`). |
| `main_filename_template` | File template for main variable inputs (e.g., `"IMERG_Hourly_Accumulation_YYYY.nc"`). |
| `env_filename_template` | File template for environmental variable inputs (e.g., `"li_YYYY.nc"`). |
| `lat_name` / `lon_name` | Names of the 1D y and x coordinate dimensions in the input files (e.g., `"lat"`, `"rlat"`). |
| `data_source` | String describing the data source (added to output metadata for traceability). |
| `dt_hours` | Temporal resolution of the data in hours (e.g., `0.5` for 30-min data, `1.0` for hourly). |

### Detection Thresholds (`detection_parameters`)
| Parameter | Example Value | Scientific Meaning |
| :--- | :--- | :--- |
| `use_env_var` | `True` / `False` | Toggle to enable/disable loading and filtering via the environmental field. |
| `core_threshold` | `10.0` | Threshold defining core convective/feature intensity (e.g., mm/h or K). |
| `envelope_threshold` | `2.0` | Threshold defining system expansion / envelope extent (e.g., mm/h or K). |
| `min_size_threshold` | `10` | Minimum size (in grid cells) required for a candidate system. |
| `min_nr_plumes` | `1` | Minimum number of core intensity plumes required within an envelope. |
| `env_var_threshold` | `-2.0` | Threshold for environmental conditioning (e.g., LI < -2 K for instability). |
| `env_var_percentage_threshold` | `0.1` | Fraction of system area (0.0 to 1.0) that must meet the environmental threshold. |

### Tracking Logic (`tracking_parameters`)
| Parameter | Default | Scientific Meaning |
| :--- | :--- | :--- |
| `main_lifetime_thresh_hours` | `4` | (hours) Minimum duration required to be classified as a mature MCS. |
| `main_area_thresh` | `3500` | (km²) Minimum system area required during the mature phase. |
| `nmaxmerge` | `5` | Maximum number of parent systems allowed to merge in a single timestep. |

### Post-Processing Filters (`postprocessing_filters`)
These filters are applied *after* tracking to remove non-MCS artifacts (e.g., synoptic fronts, erratic track jumps).

| Parameter | Example Value | Scientific Meaning |
| :--- | :--- | :--- |
| `env_var_threshold` | `1.5` | Reject systems failing lifetime environmental stability criteria (e.g., Mean LI > 1.5 K). |
| `track_straightness_threshold` | `0.4` | Reject stationary or erratic systems (0 = erratic/meandering, 1 = straight line). |
| `max_area_volatility` | `90000.0` | (km²) Reject unphysical area growth/decay spikes between consecutive timesteps. |

---

## 3. Output Data Structure

The algorithm produces **CF-compliant NetCDF4** files. Each file contains both gridded masks and tabular summary statistics.

### Gridded Variables (2D Maps)
Use these for spatial analysis and plotting.
* **`robust_mcs_id`**: The "Gold Standard". Shows the MCS **only** during its mature, active phase where size and environmental thresholds are met simultaneously. Best for climatologies.
* **`mcs_id`**: The full lifecycle (initiation $\to$ decay) of systems identified as MCSs.
* **`mcs_id_merge_split`**: The complete family tree, including all smaller convective cells that merged into or split from the main system.

### Tabular Variables (Parallel Arrays)
Use these for fast statistical analysis without loading full grids into memory.
* **`active_track_id`**: ID of the active system.
* **`active_track_lat` / `lon`**: Center of mass coordinates for each active system.
* **`active_track_touches_boundary`**: Flag (0 or 1).
    * `1`: The system touches the domain edge.
    * **Usage:** Exclude these tracks when calculating lifetime statistics to avoid domain truncation bias.

---

## 4. Best Practices for Model Evaluation

1. **Regridding:** When comparing datasets with different resolutions (e.g., RCM 12km vs CPM 3km), remap the data to a common target grid *before* tracking to ensure consistent area calculation and threshold behavior.
2. **Boundary Flag:** Always filter out tracks where `active_track_touches_boundary == 1` when analyzing total lifetime, distance, or total rainfall volume.
3. **Threshold Tuning:** Default thresholds are calibrated for midlatitude regional climate model outputs over Europe. If applying to the Tropics, US Great Plains, or satellite observations ($T_b$), adjust `core_threshold` and `envelope_threshold` accordingly, and check the resulting climatology against known diurnal/seasonal cycles.