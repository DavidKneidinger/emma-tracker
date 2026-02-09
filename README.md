 # EMMA-Tracker: Evoluation-Based Mesoscale Convective System Model Assessment-Tracker
 
 **A robust detection and tracking algorithm for Mesoscale Convective Systems (MCS), optimized for climate model evaluation.**
 ##
 Since the algorithm is not yet peer-reviewed I would kindly ask you to get in touch with me (david.kneidinger@uni-graz.at) when you plan to use it.
 ---
 
 ## Overview
 
 The **EMMA-Tracker** is a Python software designed to identify and track Mesoscale Convective Systems (MCSs) in gridded precipitation data (e.g., IMERG, CPMs, RCMs). 
 
 Unlike standard threshold-based trackers, EMMA employs a **physics-based post-processing filter** to distinguish self-sustaining, propagating MCSs from synoptic-scale frontal precipitation and stationary orographic convection. This makes it particularly suitable for process-based evaluation of regional climate models.
 
 **Key Features:**
 * **Physics-Aware Filtering:** Removes frontal systems and stationary convection using Lifted Index (LI), Track Straightness, and Area Volatility.
 * **Rotated Pole-Ready:** Output files are fully CF-compliant and support Rotated Pole grids (native EURO-CORDEX support).
 * **Robust Tracking:** Handles complex merging and splitting events explicitly.
 * **Reproducible:** Every output file embeds the full run configuration in its metadata.
 
 ## Installation
 
 We recommend using a fresh environment to ensure dependency stability.
 
 ### 1. Create Environment
 ```bash
 # Create a new environment (Python 3.10+ recommended)
 conda create -n emma python=3.11 -y
 conda activate emma
 ```
 
 ### 2. Install Package
 Clone this repository and install it in "editable" mode. This handles all dependencies (xarray, numpy, dask, etc.) automatically via `pyproject.toml`.
 
 ```bash
 git clone [https://github.com/DavidKneidinger/emma-tracker.git](https://github.com/DavidKneidinger/emma-tracker.git)
 cd emma-tracker
 pip install -e .
 ```
 
 ## Quick Start
 
 To run the tracker, you need a configuration YAML file defining your input data paths and thresholds.
 
 1.  **Configure:** Edit `config.yaml` to point to your data (see [Configuration](#-configuration) below).
 2.  **Run:** Use the command-line interface:
 
 ```bash
 emma-tracker --config config.yaml
 ```
 
 For detailed documentation on parameters and preprocessing, please see the [Technical Guide](emma/README.md).
 
 ## Configuration (`config.yaml`)
 
 The behavior of the algorithm is controlled by a single YAML file. Key parameters include:
 
 ```yaml
# config.yaml

# Data
precip_data_directory: "/path/to/hourly/precip"
lifted_index_data_directory: "/path/to/hourly/lifted_index/"
file_suffix: ".nc"
detection_output_path: "/output/detection/"
raw_tracking_output_dir: "/output/tracking_raw"  # path to the raw unfiltered tracking data
filtered_tracking_output_dir: "/output/tracking/"  # path to the final tracking data

precip_var_name: "precipitation"
liting_index_var_name: "LI"
lat_name: "rlat"
lon_name: "rlon"
data_source: "Information of your input data. This gets added to the attr of the output files"

# Years and Months to process
# empty list uses all available data in the specified directory
years: []
months: []

# Detection parameter
detection: True
use_lifted_index: True
min_size_threshold: 10  # min number of grid cells for an object to be detected
heavy_precip_threshold: 6.8  # mm/h use 99th percentile of precip product
moderate_precip_threshold: 1.0
min_nr_plumes: 1  # number of heavy precipitation plums
lifted_index_percentage_threshold: 0.1
lifted_index_threshold: -2  # K

# Tracking parameter
tracking: True
main_lifetime_thresh: 4  # min number of hours for an MCS in the mature state
main_area_thresh: 3500  # min area for a system to be in the mature state
nmaxmerge: 5  # maximum number of objects to merge in a single timestep

# Post-processing / Filtering
# This section controls the physics-based filtering logic
postprocessing: False
postprocessing_filters:
  lifted_index_threshold: 1.5          # Keep systems with mean LI < 0.0 K
  track_straightness_threshold: 0.4    # Keep systems with straightness > 0.4
  max_area_volatility: 90000.0        # Keep systems with max volatility < 0.9e5

# Other parameter
use_multiprocessing: True
number_of_cores: 20

 ```
 
 ## Output Data Structure
 
 The output is saved as **CF-compliant NetCDF4** files (compressed), organized by `YYYY/MM/`. 
 
 Each file contains both **Gridded Segmentation Masks** and **Tabular Track Statistics**:
 
 ### Gridded Variables (2D Maps)
 * `robust_mcs_id`: ID mask of MCSs during their mature, "in-phase" stage.
 * `mcs_id`: Full lifecycle mask of Main MCSs.
 * `mcs_id_merge_split`: Complete family tree mask (includes mergers/splits).
 
 ### Tabular Variables (Parallel Arrays)
 For efficient analysis without loading full grids, summary statistics are stored as parallel arrays along the `tracks` dimension:
 * `active_track_id`: The ID of the system.
 * `active_track_lat` / `lon`: The precipitation-weighted center of mass.
 * `active_track_touches_boundary`: Flag (`1`) if the system touches the domain edge (crucial for lifetime statistics).
 
 ## Testing & Validation
 
 This repository includes a regression test suite that generates synthetic convective scenarios (growth, merger, splitting) to verify algorithmic stability.
 
 To run the tests:
 ```bash
 pip install pytest
 pytest
 ```
 
 ## Citation
 If you use the EMMA-Tracker in your research, please cite the following paper (currently still preprint):
 
 https://doi.org/10.22541/essoar.176798036.66459300/v2

 ## Reference Dataset
 A European 27-year warm-season dataset of MCSs, based on IMERG precipitation and ERA5 derived lifted index:
 https://zenodo.org/records/18234276 

 
 ## Improvments and suggestions
 If you encounter any errors or have suggestions for improvement, simply contact me or open um an issue directly in github.

 ## License
 
 This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
 
 ---
 **Contact:** David Kneidinger (david.kneidinger@uni-graz.at)  
 *Wegener Center for Climate and Global Change, University of Graz*

 **Funding:**
 Development of this algorithm was funded by the Österreichische Forschungsförderungsgesellschaft (FFG) as part of the MoCCA project.
