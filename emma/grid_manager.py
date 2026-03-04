import numpy as np
import logging
import xarray as xr
from pyproj import CRS, Geod

logger = logging.getLogger(__name__)


def get_attr_case_insensitive(obj, target_attr):
    """
    Returns the value of an attribute regardless of its case (e.g., 'gridType' vs 'gridtype').

    Parameters:
    - obj: xarray Dataset or DataArray object containing attributes.
    - target_attr: String representing the target attribute name to find.

    Returns:
    - The attribute value if found, otherwise None.
    """
    for attr_name in obj.attrs:
        if attr_name.lower() == target_attr.lower():
            return obj.attrs[attr_name]
    return None


def extract_cf_metadata(ds, lat_name, lon_name):
    """
    Extracts grid mapping metadata.    
    Instead of using pyproj to generate a modernized CF-1.8 dictionary (which 
    can break older visualization tools like ncview), this function strictly 
    preserves the original metadata structure of the input file to maintain 
    provenance and compatibility. It only intervenes to correct missing essential 
    CF fields and strip known bloated attributes (like crs_wkt) added by 
    intermediate remapping tools.

    Parameters:
    - ds: xarray Dataset containing the input data.
    - lat_name: Name of the latitude/y-dimension variable.
    - lon_name: Name of the longitude/x-dimension variable.

    Returns:
    - cf_dict: A dictionary of projection attributes, including a special 
               '__var_name__' key to remember the original dummy variable's name.
    """
    # 1. Check for existing CF-compliant 'grid_mapping' variable
    target_vars = list(ds.data_vars) + [lat_name, lon_name, 'latitude', 'longitude', 'lat', 'lon']
    for v_name in target_vars:
        if v_name not in ds: continue
        mapping_var_name = get_attr_case_insensitive(ds[v_name], 'grid_mapping')
        
        if mapping_var_name and mapping_var_name in ds:
            cf_dict = ds[mapping_var_name].attrs.copy()
            
            # --- THE "IN-BETWEEN" SMART PATCHING ---
            # 1. Ensure grid_mapping_name exists (Patch missing data)
            if 'grid_mapping_name' not in cf_dict:
                if 'grid_north_pole_latitude' in cf_dict:
                    cf_dict['grid_mapping_name'] = 'rotated_latitude_longitude'
                else:
                    cf_dict['grid_mapping_name'] = 'unknown'
            
            # 2. Strip GDAL/pyproj/CDO bloat if it was accidentally inherited
            keys_to_remove = [
                "crs_wkt", "semi_major_axis", "semi_minor_axis", "inverse_flattening",
                "reference_ellipsoid_name", "longitude_of_prime_meridian",
                "prime_meridian_name", "geographic_crs_name", "horizontal_datum_name"
            ]
            for k in keys_to_remove:
                cf_dict.pop(k, None)
            
            # 3. Store the original variable name (e.g., "rotated_pole") so the saver can use it
            cf_dict["__var_name__"] = str(mapping_var_name)
            
            return cf_dict

    # 2. Fallback: Search for GRIB-specific grid type tags and translate
    search_objs = [ds] + [ds[v] for v in ds.data_vars]
    for obj in search_objs:
        grib_type = get_attr_case_insensitive(obj, 'GRIB_gridType')
        if grib_type:
            if grib_type.lower() == 'lambert':
                lat_1 = get_attr_case_insensitive(obj, 'GRIB_Latin1InDegrees') or 50.0
                lat_2 = get_attr_case_insensitive(obj, 'GRIB_Latin2InDegrees') or 50.0
                lat_0 = get_attr_case_insensitive(obj, 'GRIB_LaDInDegrees') or 50.0
                lon_0 = get_attr_case_insensitive(obj, 'GRIB_LoVInDegrees') or 8.0
                
                return {
                    "__var_name__": "lambert_conformal",
                    "grid_mapping_name": "lambert_conformal_conic",
                    "standard_parallel": [lat_1, lat_2],
                    "latitude_of_projection_origin": lat_0,
                    "longitude_of_central_meridian": lon_0
                }
            else:
                logger.warning(f"Unhandled GRIB grid type: {grib_type}.")
                return {'__var_name__': 'crs', 'grid_mapping_name': grib_type}

    # 3. Last Resort Inference
    if "rlat" in lat_name.lower() or "rlon" in lon_name.lower():
        return {'__var_name__': 'rotated_pole', 'grid_mapping_name': 'rotated_latitude_longitude'}
        
    return {'__var_name__': 'crs', 'grid_mapping_name': 'latitude_longitude'}


def compute_grid_area(lat2d, lon2d):
    """
    Calculates the exact physical area (in km^2) of every individual grid cell
    using the WGS84 ellipsoid. This completely eliminates projection distortion errors.

    Parameters:
    - lat2d: 2D numpy array of true geographic latitudes.
    - lon2d: 2D numpy array of true geographic longitudes.

    Returns:
    - area_map: 2D numpy array of the same shape as lat2d, containing the area of each cell in km^2.
    """
    logger.info("Initializing precise WGS84 ellipsoidal area map...")
    geod = Geod(ellps="WGS84")

    # Calculate pixel width (dx) by measuring distance to the Eastern neighbor
    _, _, dx = geod.inv(
        lon2d[:, :-1].flatten(),
        lat2d[:, :-1].flatten(),
        lon2d[:, 1:].flatten(),
        lat2d[:, 1:].flatten(),
    )
    dx = dx.reshape(lon2d[:, :-1].shape)

    # Calculate pixel height (dy) by measuring distance to the Northern neighbor
    _, _, dy = geod.inv(
        lon2d[:-1, :].flatten(),
        lat2d[:-1, :].flatten(),
        lon2d[1:, :].flatten(),
        lat2d[1:, :].flatten(),
    )
    dy = dy.reshape(lon2d[:-1, :].shape)

    # Pad the arrays to match the original grid shape (copying edge values)
    dx_full = np.pad(dx, ((0, 0), (0, 1)), mode="edge")
    dy_full = np.pad(dy, ((0, 1), (0, 0)), mode="edge")

    # Calculate area in km^2
    area_map = (dx_full / 1000.0) * (dy_full / 1000.0)

    return area_map


def build_grid_info(ds, lat_name, lon_name, lat2d, lon2d):
    """
    Packages all structural coordinates, geographic coordinates, precise area calculations,
    and CF-compliant projection metadata into a single transfer dictionary.

    This function should be called ONLY ONCE per dataset stream to initialize the grid template.

    Parameters:
    - ds: xarray Dataset of the master file.
    - lat_name: Name of the 1D y-dimension.
    - lon_name: Name of the 1D x-dimension.
    - lat2d: 2D numpy array of true geographic latitudes.
    - lon2d: 2D numpy array of true geographic longitudes.

    Returns:
    - grid_info: Dictionary containing all static spatial properties of the grid.
    """
    logger.info("Building global grid template...")

    # Extract CF compliant metadata
    cf_metadata = extract_cf_metadata(ds, lat_name, lon_name)

    # Calculate static area map
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
    }

    logger.info(
        f"Grid template built. Identified CF Mapping: {cf_metadata.get('grid_mapping_name', 'Unknown')}"
    )
    return grid_info


def verify_and_build_grid_template(
    first_precip_file, first_li_file, y_dim_name, x_dim_name
):
    """
    Performs STRICT initial grid validation by comparing the spatial coordinates
    of the first Precipitation and Lifted Index files. If they match bit-for-bit,
    it builds and returns the global grid template containing CF-metadata and
    the exact ellipsoidal area map.

    Parameters:
    - first_precip_file (str): Path to the first precipitation NetCDF file.
    - first_li_file (str or None): Path to the first lifted index NetCDF file (if used).
    - y_dim_name (str): Name of the 1D y-dimension (from config).
    - x_dim_name (str): Name of the 1D x-dimension (from config).

    Returns:
    - global_grid_template (dict): The verified spatial grid template.
    """
    logger.info("Performing STRICT initial grid validation and building template...")

    with xr.open_dataset(first_precip_file, engine="netcdf4") as ds_p:
        p_lat2d = ds_p["latitude"].values if "latitude" in ds_p else None
        p_lon2d = ds_p["longitude"].values if "longitude" in ds_p else None

        if p_lat2d is None:
            # Fallback for regular grids
            p_lon2d, p_lat2d = np.meshgrid(
                ds_p[x_dim_name].values, ds_p[y_dim_name].values
            )

        if first_li_file:
            with xr.open_dataset(first_li_file, engine="netcdf4") as ds_l:
                l_lat2d = ds_l["latitude"].values if "latitude" in ds_l else None
                l_lon2d = ds_l["longitude"].values if "longitude" in ds_l else None

                if l_lat2d is None:
                    l_lon2d, l_lat2d = np.meshgrid(
                        ds_l[x_dim_name].values, ds_l[y_dim_name].values
                    )

                # Head Developer Check: Verify BOTH latitude and longitude
                if not np.array_equal(p_lat2d, l_lat2d) or not np.array_equal(
                    p_lon2d, l_lon2d
                ):
                    logger.critical(
                        "CRITICAL GRID MISMATCH: Precip and LI spatial coordinates differ bit-for-bit."
                    )
                    sys.exit(1)

        # Build the global template
        global_grid_template = build_grid_info(
            ds_p, y_dim_name, x_dim_name, p_lat2d, p_lon2d
        )

    return global_grid_template
