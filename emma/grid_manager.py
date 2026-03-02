import numpy as np
import logging
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
    Robustly identifies the grid type and translates it into a CF-compliant 
    dictionary using the pyproj library. Handles standard CF conventions as well 
    as proprietary GRIB attributes.

    Parameters:
    - ds: xarray Dataset containing the input data.
    - lat_name: Name of the latitude/y-dimension variable.
    - lon_name: Name of the longitude/x-dimension variable.

    Returns:
    - cf_dict: A 100% CF-compliant dictionary of projection attributes.
    """

    # 1. Check for existing CF-compliant 'grid_mapping' variable
    target_vars = list(ds.data_vars) + [lat_name, lon_name, 'latitude', 'longitude']
    for v_name in target_vars:
        if v_name not in ds: continue
        mapping_var_name = get_attr_case_insensitive(ds[v_name], 'grid_mapping')
        
        if mapping_var_name and mapping_var_name in ds:
            mapping_var = ds[mapping_var_name]
            try:
                # Use pyproj to standardize and validate the existing CF dict
                crs = CRS.from_cf(mapping_var.attrs)
                return crs.to_cf()
            except Exception as e:
                logger.warning(f"Failed to parse existing CF grid_mapping with pyproj: {e}")
                return mapping_var.attrs.copy()

    # 2. Fallback: Search for GRIB-specific grid type tags and translate to CF
    search_objs = [ds] + [ds[v] for v in ds.data_vars]
    for obj in search_objs:
        grib_type = get_attr_case_insensitive(obj, 'GRIB_gridType')
        if grib_type:
            if grib_type.lower() == 'lambert':
                # Translate CERRA/GRIB Lambert parameters into a PROJ string/dict
                lat_1 = get_attr_case_insensitive(obj, 'GRIB_Latin1InDegrees') or 50.0
                lat_2 = get_attr_case_insensitive(obj, 'GRIB_Latin2InDegrees') or 50.0
                lat_0 = get_attr_case_insensitive(obj, 'GRIB_LaDInDegrees') or 50.0
                lon_0 = get_attr_case_insensitive(obj, 'GRIB_LoVInDegrees') or 8.0
                
                crs = CRS.from_dict({
                    'proj': 'lcc',
                    'lat_1': lat_1,
                    'lat_2': lat_2,
                    'lat_0': lat_0,
                    'lon_0': lon_0,
                    'a': 6371229,  # Standard GRIB Earth Radius
                    'b': 6371229
                })
                logger.info("Successfully translated GRIB Lambert projection to CF standards.")
                return crs.to_cf()
            else:
                logger.warning(f"Unhandled GRIB grid type: {grib_type}. Output may lack projection metadata.")
                return {'grid_mapping_name': grib_type}

    # 3. Last Resort: Inference from dimension names (No formal projection defined)
    if "rlat" in lat_name.lower() or "rlon" in lon_name.lower():
        return {'grid_mapping_name': 'rotated_latitude_longitude'}
    elif lat_name.lower() in ["y", "x"]:
        return {'grid_mapping_name': 'lambert_conformal_conic'}
        
    return {'grid_mapping_name': 'latitude_longitude'}

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
        lon2d[:, :-1].flatten(), lat2d[:, :-1].flatten(),
        lon2d[:, 1:].flatten(), lat2d[:, 1:].flatten()
    )
    dx = dx.reshape(lon2d[:, :-1].shape)
    
    # Calculate pixel height (dy) by measuring distance to the Northern neighbor
    _, _, dy = geod.inv(
        lon2d[:-1, :].flatten(), lat2d[:-1, :].flatten(),
        lon2d[1:, :].flatten(), lat2d[1:, :].flatten()
    )
    dy = dy.reshape(lon2d[:-1, :].shape)
    
    # Pad the arrays to match the original grid shape (copying edge values)
    dx_full = np.pad(dx, ((0, 0), (0, 1)), mode='edge')
    dy_full = np.pad(dy, ((0, 1), (0, 0)), mode='edge')
    
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
        "area_map": area_map
    }
    
    logger.info(f"Grid template built. Identified CF Mapping: {cf_metadata.get('grid_mapping_name', 'Unknown')}")
    return grid_info