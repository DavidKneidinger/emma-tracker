import numpy as np
import logging
import xarray as xr
from pyproj import CRS, Geod
import sys

logger = logging.getLogger(__name__)


def get_attr_case_insensitive(obj, target_attr):
    """Returns the value of an attribute regardless of its case."""
    for attr_name in obj.attrs:
        if attr_name.lower() == target_attr.lower():
            return obj.attrs[attr_name]
    return None


def extract_cf_metadata(ds, lat_name, lon_name):
    """Extracts grid mapping metadata strictly preserving original file structure."""
    target_vars = list(ds.data_vars) + [
        lat_name,
        lon_name,
        "latitude",
        "longitude",
        "lat",
        "lon",
    ]
    for v_name in target_vars:
        if v_name not in ds:
            continue
        mapping_var_name = get_attr_case_insensitive(ds[v_name], "grid_mapping")

        if mapping_var_name and mapping_var_name in ds:
            cf_dict = ds[mapping_var_name].attrs.copy()

            if "grid_mapping_name" not in cf_dict:
                if "grid_north_pole_latitude" in cf_dict:
                    cf_dict["grid_mapping_name"] = "rotated_latitude_longitude"
                else:
                    cf_dict["grid_mapping_name"] = "unknown"

            keys_to_remove = [
                "crs_wkt",
                "semi_major_axis",
                "semi_minor_axis",
                "inverse_flattening",
                "reference_ellipsoid_name",
                "longitude_of_prime_meridian",
                "prime_meridian_name",
                "geographic_crs_name",
                "horizontal_datum_name",
            ]
            for k in keys_to_remove:
                cf_dict.pop(k, None)

            cf_dict["__var_name__"] = str(mapping_var_name)
            return cf_dict

    search_objs = [ds] + [ds[v] for v in ds.data_vars]
    for obj in search_objs:
        grib_type = get_attr_case_insensitive(obj, "GRIB_gridType")
        if grib_type:
            if grib_type.lower() == "lambert":
                lat_1 = get_attr_case_insensitive(obj, "GRIB_Latin1InDegrees") or 50.0
                lat_2 = get_attr_case_insensitive(obj, "GRIB_Latin2InDegrees") or 50.0
                lat_0 = get_attr_case_insensitive(obj, "GRIB_LaDInDegrees") or 50.0
                lon_0 = get_attr_case_insensitive(obj, "GRIB_LoVInDegrees") or 8.0

                return {
                    "__var_name__": "lambert_conformal",
                    "grid_mapping_name": "lambert_conformal_conic",
                    "standard_parallel": [lat_1, lat_2],
                    "latitude_of_projection_origin": lat_0,
                    "longitude_of_central_meridian": lon_0,
                }
            else:
                logger.warning(f"Unhandled GRIB grid type: {grib_type}.")
                return {"__var_name__": "crs", "grid_mapping_name": grib_type}

    if "rlat" in lat_name.lower() or "rlon" in lon_name.lower():
        return {
            "__var_name__": "rotated_pole",
            "grid_mapping_name": "rotated_latitude_longitude",
        }

    return {"__var_name__": "crs", "grid_mapping_name": "latitude_longitude"}


def compute_grid_area(lat2d, lon2d):
    """Calculates cell area in km^2 using WGS84 ellipsoid."""
    logger.info("Initializing precise WGS84 ellipsoidal area map...")
    geod = Geod(ellps="WGS84")

    _, _, dx = geod.inv(
        lon2d[:, :-1].flatten(),
        lat2d[:, :-1].flatten(),
        lon2d[:, 1:].flatten(),
        lat2d[:, 1:].flatten(),
    )
    dx = dx.reshape(lon2d[:, :-1].shape)

    _, _, dy = geod.inv(
        lon2d[:-1, :].flatten(),
        lat2d[:-1, :].flatten(),
        lon2d[1:, :].flatten(),
        lat2d[1:, :].flatten(),
    )
    dy = dy.reshape(lon2d[:-1, :].shape)

    dx_full = np.pad(dx, ((0, 0), (0, 1)), mode="edge")
    dy_full = np.pad(dy, ((0, 1), (0, 0)), mode="edge")

    area_map = (dx_full / 1000.0) * (dy_full / 1000.0)
    return area_map


def build_grid_info(
    ds,
    lat_name,
    lon_name,
    lat2d,
    lon2d,
    main_var_units="unknown",
    env_var_units="unknown",
):
    """Packages grid coordinates, area map, and projection metadata into a dictionary."""
    logger.info("Building global grid template...")

    cf_metadata = extract_cf_metadata(ds, lat_name, lon_name)
    area_map = compute_grid_area(lat2d, lon2d)

    grid_info = {
        "y_dim_name": lat_name,
        "x_dim_name": lon_name,
        "lat1d": ds[lat_name].values,
        "lon1d": ds[lon_name].values,
        "lat2d": lat2d,
        "lon2d": lon2d,
        "y_attrs": ds[lat_name].attrs.copy(),
        "x_attrs": ds[lon_name].attrs.copy(),
        "cf_metadata": cf_metadata,
        "area_map": area_map,
        "main_var_units": main_var_units,
        "env_var_units": env_var_units,
    }

    logger.info(
        f"Grid template built. Identified CF Mapping: {cf_metadata.get('grid_mapping_name', 'Unknown')}"
    )
    return grid_info


def verify_and_build_grid_template(
    first_main_var_file,
    first_env_var_file,
    y_dim_name,
    x_dim_name,
    main_var_name=None,
    env_var_name=None,
):
    """
    Performs initial grid validation and dynamically extracts variable units from input datasets.
    """
    logger.info("Performing STRICT initial grid validation and building template...")

    with xr.open_dataset(first_main_var_file, engine="netcdf4") as ds_m:

        if y_dim_name not in ds_m.sizes or x_dim_name not in ds_m.sizes:
            msg = (
                f"CRITICAL ERROR: Configured dimensions '{y_dim_name}' or '{x_dim_name}' "
                f"are not recognized. Available dimensions: {list(ds_m.sizes.keys())}"
            )
            logger.critical(msg)
            print(f"\n{msg}\n")
            sys.exit(1)

        if ds_m[y_dim_name].ndim != 1 or ds_m[x_dim_name].ndim != 1:
            msg = (
                f"CRITICAL ERROR: '{y_dim_name}' and '{x_dim_name}' must be 1D arrays."
            )
            logger.critical(msg)
            print(f"\n{msg}\n")
            sys.exit(1)

        # Dynamically extract main variable units
        main_var_units = "unknown"
        if main_var_name and main_var_name in ds_m:
            main_var_units = str(ds_m[main_var_name].attrs.get("units", "unknown"))

        m_lat2d, m_lon2d = None, None
        for lat_candidate in ["latitude", "lat"]:
            if lat_candidate in ds_m and ds_m[lat_candidate].ndim == 2:
                m_lat2d = ds_m[lat_candidate].values
                break

        for lon_candidate in ["longitude", "lon"]:
            if lon_candidate in ds_m and ds_m[lon_candidate].ndim == 2:
                m_lon2d = ds_m[lon_candidate].values
                break

        if m_lat2d is None or m_lon2d is None:
            m_lon2d, m_lat2d = np.meshgrid(
                ds_m[x_dim_name].values, ds_m[y_dim_name].values
            )

        env_var_units = "unknown"
        if first_env_var_file:
            with xr.open_dataset(first_env_var_file, engine="netcdf4") as ds_e:
                if env_var_name and env_var_name in ds_e:
                    env_var_units = str(
                        ds_e[env_var_name].attrs.get("units", "unknown")
                    )

                e_lat2d, e_lon2d = None, None
                for lat_candidate in ["latitude", "lat"]:
                    if lat_candidate in ds_e and ds_e[lat_candidate].ndim == 2:
                        e_lat2d = ds_e[lat_candidate].values
                        break

                for lon_candidate in ["longitude", "lon"]:
                    if lon_candidate in ds_e and ds_e[lon_candidate].ndim == 2:
                        e_lon2d = ds_e[lon_candidate].values
                        break

                if e_lat2d is None or e_lon2d is None:
                    e_lon2d, e_lat2d = np.meshgrid(
                        ds_e[x_dim_name].values, ds_e[y_dim_name].values
                    )

                lat_match = np.allclose(m_lat2d, e_lat2d, atol=1e-4, equal_nan=True)
                lon_match = np.allclose(m_lon2d, e_lon2d, atol=1e-4, equal_nan=True)

                if not lat_match or not lon_match:
                    msg = "CRITICAL GRID MISMATCH: Main var and Env var spatial coordinates differ."
                    logger.critical(msg)
                    print(f"\n{msg}\n")
                    sys.exit(1)

        global_grid_template = build_grid_info(
            ds_m,
            y_dim_name,
            x_dim_name,
            m_lat2d,
            m_lon2d,
            main_var_units=main_var_units,
            env_var_units=env_var_units,
        )

    return global_grid_template
