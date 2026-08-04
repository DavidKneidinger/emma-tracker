import yaml
import os
from dataclasses import dataclass, field
from typing import List, Optional

VALID_OPERATORS = {">=", "<=", ">", "<"}


def validate_operator(op_str: str, param_name: str):
    if op_str not in VALID_OPERATORS:
        raise ValueError(
            f"❌ CONFIG ERROR: Invalid '{param_name}' value '{op_str}'. Must be one of {sorted(list(VALID_OPERATORS))}."
        )


# --- NESTED SECTIONS ---


@dataclass
class DetectionParameters:
    use_env_var: bool
    main_var_operator: str
    env_var_operator: str
    min_size_threshold: int
    core_threshold: float
    envelope_threshold: float
    min_nr_plumes: int
    env_var_percentage_threshold: float
    env_var_threshold: float


@dataclass
class TrackingParameters:
    main_lifetime_thresh_hours: int
    main_area_thresh: float
    nmaxmerge: int


@dataclass
class PostProcessingFilters:
    env_var_operator: str
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

    def get_active_metadata(self) -> dict:
        """
        Returns a flat dictionary containing only the active configuration options,
        filepaths, and thresholds used in the current run. Disabled phases/features
        are completely omitted. Booleans are converted to integers for NetCDF compatibility.
        """
        use_env = self.detection_parameters.use_env_var

        meta = {
            "data_source": str(self.data_source),
            "dt_hours": float(self.dt_hours),
            "lat_name": str(self.lat_name),
            "lon_name": str(self.lon_name),
            "main_var_name": str(self.main_var_name),
            "main_var_data_directory": str(self.main_var_data_directory),
            "main_var_filename_template": str(self.main_var_filename_template),
            "use_env_var": int(use_env),
            "detection_enabled": int(self.detection),
            "tracking_enabled": int(self.tracking),
            "postprocessing_enabled": int(self.postprocessing),
        }

        # Include environmental paths only if enabled
        if use_env:
            meta["env_var_name"] = str(self.env_var_name)
            meta["env_var_data_directory"] = str(self.env_var_data_directory)
            meta["env_var_filename_template"] = str(self.env_var_filename_template)

        # Detection Thresholds & Operators
        if self.detection:
            meta["det_main_var_operator"] = str(
                self.detection_parameters.main_var_operator
            )
            meta["det_min_size_threshold"] = int(
                self.detection_parameters.min_size_threshold
            )
            meta["det_core_threshold"] = float(self.detection_parameters.core_threshold)
            meta["det_envelope_threshold"] = float(
                self.detection_parameters.envelope_threshold
            )
            meta["det_min_nr_plumes"] = int(self.detection_parameters.min_nr_plumes)
            if use_env:
                meta["det_env_var_operator"] = str(
                    self.detection_parameters.env_var_operator
                )
                meta["det_env_var_threshold"] = float(
                    self.detection_parameters.env_var_threshold
                )
                meta["det_env_var_percentage_threshold"] = float(
                    self.detection_parameters.env_var_percentage_threshold
                )

        # Tracking Thresholds
        if self.tracking:
            meta["track_main_lifetime_thresh_hours"] = int(
                self.tracking_parameters.main_lifetime_thresh_hours
            )
            meta["track_main_area_thresh"] = float(
                self.tracking_parameters.main_area_thresh
            )
            meta["track_nmaxmerge"] = int(self.tracking_parameters.nmaxmerge)

        # Postprocessing Filters & Operators
        if self.postprocessing:
            meta["pp_track_straightness_threshold"] = float(
                self.postprocessing_filters.track_straightness_threshold
            )
            meta["pp_max_area_volatility"] = float(
                self.postprocessing_filters.max_area_volatility
            )
            if use_env:
                meta["pp_env_var_operator"] = str(
                    self.postprocessing_filters.env_var_operator
                )
                meta["pp_env_var_threshold"] = float(
                    self.postprocessing_filters.env_var_threshold
                )

        return meta

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

            def pop_section(key, dataclass_type):
                section_data = data.pop(key)
                if section_data is None:
                    raise KeyError(f"Section '{key}' is empty or missing.")
                return dataclass_type(**section_data)

            det_params = pop_section("detection_parameters", DetectionParameters)
            track_params = pop_section("tracking_parameters", TrackingParameters)
            pp_filters = pop_section("postprocessing_filters", PostProcessingFilters)

            # Validate operators
            validate_operator(
                det_params.main_var_operator, "detection_parameters.main_var_operator"
            )
            validate_operator(
                det_params.env_var_operator, "detection_parameters.env_var_operator"
            )
            validate_operator(
                pp_filters.env_var_operator, "postprocessing_filters.env_var_operator"
            )

            return cls(
                detection_parameters=det_params,
                tracking_parameters=track_params,
                postprocessing_filters=pp_filters,
                **data,
            )

        except KeyError as e:
            raise KeyError(
                f"❌ CONFIG ERROR: Missing required key or section in {path}: {e}"
            )
        except TypeError as e:
            raise TypeError(
                f"❌ CONFIG ERROR: Invalid key or type mismatch in {path}. Detail: {e}"
            )
