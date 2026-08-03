import yaml
import os
from dataclasses import dataclass, field
from typing import List, Optional

# --- NESTED SECTIONS ---


@dataclass
class DetectionParameters:  # Fixed typo: Paramters -> Parameters
    use_env_var: bool
    min_size_threshold: int
    core_threshold: float
    envelope_threshold: float
    min_nr_plumes: int
    env_var_percentage_threshold: float
    env_var_threshold: float


@dataclass
class TrackingParameters:  # Fixed typo: Paramters -> Parameters
    main_lifetime_thresh_hours: int
    main_area_thresh: float
    nmaxmerge: int


@dataclass
class PostProcessingFilters:
    env_var_threshold: float
    track_straightness_threshold: float
    max_area_volatility: float


# --- MAIN CONFIG ---


@dataclass
class EmmaConfig:
    # 1. Paths
    main_var_data_directory: str
    env_var_data_directory: str
    detection_output_path: str
    raw_tracking_output_dir: str
    filtered_tracking_output_dir: str

    # 2. Variable Names
    main_var_name: str
    env_var_name: str
    main_var_filename_template: str
    env_var_filename_template: str
    lat_name: str
    lon_name: str
    data_source: str

    dt_hours: float

    # 3. Selection
    years: List[int]
    months: List[int]

    # 4. Toggles
    detection: bool
    tracking: bool
    postprocessing: bool

    # 5. Nested Configs
    detection_parameters: DetectionParameters
    tracking_parameters: TrackingParameters
    postprocessing_filters: PostProcessingFilters

    # 6. System
    use_multiprocessing: bool
    number_of_cores: int

    @classmethod
    def load(cls, path: str) -> "EmmaConfig":
        """
        Loads the YAML file, validates strict types/keys, and returns the Config object.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Config file not found: {path}")

        with open(path, "r") as f:
            data = yaml.safe_load(f)

        try:
            # Helper to extract and remove a section from the dict
            def pop_section(key, dataclass_type):
                section_data = data.pop(key)
                if section_data is None:
                    raise KeyError(f"Section '{key}' is empty or missing.")
                return dataclass_type(**section_data)

            # Build nested objects first
            # These keys MUST match your YAML exactly
            det_params = pop_section("detection_parameters", DetectionParameters)
            track_params = pop_section("tracking_parameters", TrackingParameters)
            pp_filters = pop_section("postprocessing_filters", PostProcessingFilters)

            # Build main object
            return cls(
                detection_parameters=det_params,
                tracking_parameters=track_params,
                postprocessing_filters=pp_filters,
                **data,  # Unpacks the rest of the flat keys
            )

        except KeyError as e:
            raise KeyError(
                f"❌ CONFIG ERROR: Missing required key or section in {path}: {e}"
            )
        except TypeError as e:
            # This catches extra keys or wrong types
            raise TypeError(
                f"❌ CONFIG ERROR: Invalid key or type mismatch in {path}. Detail: {e}"
            )
