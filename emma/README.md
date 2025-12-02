 # EMMA-Tracker: User Guide & Technical Documentation
 
 This document provides detailed instructions on data preprocessing, configuration parameters, and the scientific logic behind the detection and tracking algorithm.
 
 ---
 
 ## 1. Data Preprocessing
 
 The EMMA-Tracker requires two primary input fields: **Precipitation** and **Lifted Index (LI)**.
 
 ### Precipitation
 * **Format:** NetCDF (.nc).
 * **Resolution:** The algorithm is tuned for high-resolution data (approx. 0.1° or 12km and finer).
 * **Units:** mm/h (or kg m-2 s-1, the tool handles conversion).
 
 ### Atmospheric Instability (Lifted Index)
 To use the physics-based filtering, you must provide the Lifted Index (LI).
 * **Source:** Typically derived from ERA5 or Model Level data.
 * **Calculation:** Scripts to calculate LI from standard pressure levels (T, Q, Z) are located in the `../preprocess/` directory.
 * **Grid:** Must match the precipitation grid exactly.
 
 **Note:** If your model output does not have LI, you can run the tracker with `use_lifted_index: False`, though this reduces the physical robustness of the resulting climatology.
 
 ---
 
 ## 2. Configuration Guide (`config.yaml`)
 
 The algorithm is controlled by a single YAML file. Below is a detailed explanation of the parameters.
 
 ### Data & Paths
 | Parameter | Description |
 | :--- | :--- |
 | `precip_data_directory` | Root folder containing input NetCDF files. |
 | `raw_tracking_output_dir` | Where the initial tracking results (unfiltered) are saved. |
 | `filtered_tracking_output_dir` | Where the final, publication-ready MCS tracks are saved. |
 | `file_suffix` | Pattern to match input files (e.g., `.nc`). |
 
 ### Detection Thresholds
 | Parameter | Default | Scientific Meaning |
 | :--- | :--- | :--- |
 | `heavy_precip_threshold` | 6.82 | (mm/h) Defines the convective core. Usually the 99th percentile of wet hours. |
 | `moderate_precip_threshold` | 1.0 | (mm/h) Defines the system extent (morphological dilation). |
 | `min_size_threshold` | 10 | (number of grid cells) Minimum size to be considered a candidate. |
 
 ### Tracking Logic
 | Parameter | Default | Scientific Meaning |
 | :--- | :--- | :--- |
 | `main_lifetime_thresh` | 4 | (hours) Minimum duration to be classified as an MCS. |
 | `main_area_thresh` | 5000 | (km²) Minimum area required during the mature phase. |
 | `nmaxmerge` | 5 | Max number of systems allowed to merge in one timestep. |
 
 ### Post-Processing Filters (The Physics Check)
 These filters are applied *after* tracking to remove non-MCS artifacts (fronts, orographic noise).
 
 ```yaml
 run_postprocessing: True
 postprocessing_filters:
   lifted_index_threshold: 0.0       # Reject systems in stable environments (Mean LI > 0 K)
   track_straightness_threshold: 0.4 # Reject stationary/erratic systems (0 = erratic, 1 = straight)
   max_area_volatility: 120000.0     # Reject unphysical growth spikes (e.g., frontal mergers)
 ```
 
 ---
 
 ## 3. Output Data Structure
 
 The algorithm produces **CF-compliant NetCDF4** files. Each file contains both gridded masks and tabular summary statistics.
 
 ### Gridded Variables (2D Maps)
 Use these for spatial analysis and plotting.
 * **`robust_mcs_id`**: The "Gold Standard". Shows the MCS **only** during its mature, unstable phase. Best for climatologies.
 * **`mcs_id`**: The full lifecycle (initiation $\to$ decay) of systems identified as MCSs.
 * **`mcs_id_merge_split`**: The complete family tree, including all small convective cells that merged into the main system.
 
 ### Tabular Variables (Parallel Arrays)
 Use these for fast statistical analysis without loading full grids.
 * **`active_track_id`**: ID of the system.
 * **`active_track_lat` / `lon`**: Precipitation-weighted center of mass.
 * **`active_track_touches_boundary`**: Flag (0 or 1).
     * `1`: The system touches the domain edge.
     * **Usage:** Exclude these tracks when calculating lifetime statistics to avoid bias.
 
 ---
 
 ## 4. Best Practices for Model Evaluation
 
 1.  **Regridding:** When comparing models with different resolutions (e.g., RCM 12km vs CPM 3km), regrid the precipitation to a common grid *before* tracking to ensure fair area comparisons.
 2.  **Boundary Flag:** Always use the `active_track_touches_boundary` flag to filter out incomplete tracks when analyzing duration.
 3.  **Tuning:** The default thresholds are tuned for Continental Europe. If applying to the Tropics or US Great Plains, recalculate the `heavy_precip_threshold` (P99).