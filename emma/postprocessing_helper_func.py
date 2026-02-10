"""
emma/postprocessing_helper_func.py

Helper functions for MCS track analysis and physical property calculations.
Includes:
- Grid area calculation
- Kinematics (distance, displacement, straightness)
- Area volatility analysis
"""

import numpy as np
import pandas as pd
import math
from geopy.distance import great_circle

R_EARTH_KM = 6371.0

def _haversine_vec(lat1_deg, lon1_deg, lat2_deg, lon2_deg):
    """Vectorized great-circle distance (km)."""
    lat1, lon1, lat2, lon2 = np.radians([lat1_deg, lon1_deg, lat2_deg, lon2_deg])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return R_EARTH_KM * c


def calculate_grid_area_map(detection_result):
    """Calculates the precise area (km²) of each cell in the grid."""

    lat_1d = detection_result['lat_1d'] 
    lat_2d = detection_result['lat2d']
    lon_2d = detection_result['lon2d']
        
    grid_shape = lat_2d.shape
    (n_lats, n_lons) = grid_shape
    
    # Check for Regular Grid (IMERG-style 1D lat/lon)
    is_regular = False
    if lat_1d is not None and lat_1d.shape[0] == n_lats:
         try:
            is_regular = np.allclose(lat_2d, lat_1d[:, np.newaxis])
         except ValueError:
            is_regular = False

    if is_regular:
        # --- Regular Grid Calculation (IMERG) ---
        lon_1d = detection_result['lon_1d']
        if len(lat_1d) > 1:
            delta_lat_deg = np.abs(np.diff(lat_1d)[0])
            delta_lon_deg = np.abs(np.diff(lon_1d)[0])
        else:
            delta_lat_deg = delta_lon_deg = 0.1

        lat_rad_1d = np.radians(lat_1d)
        delta_lat_rad = math.radians(delta_lat_deg)
        delta_lon_rad = math.radians(delta_lon_deg)
        
        north_south_dist_km = R_EARTH_KM * delta_lat_rad
        east_west_dist_km_1d = R_EARTH_KM * np.cos(lat_rad_1d) * delta_lon_rad
        
        area_1d = east_west_dist_km_1d * north_south_dist_km
        area_column = area_1d[:, np.newaxis]
        return np.tile(area_column, (1, n_lons))
    
    else:
        # --- Irregular/Rotated Grid (CORDEX) - EXACT METHOD ---
        # 1. Calculate dy (North-South distance) - Constant in rotated grid
        i_mid, j_mid = n_lats // 2, n_lons // 2
        dy_km = _haversine_vec(
            lat_2d[i_mid, j_mid], lon_2d[i_mid, j_mid],
            lat_2d[i_mid+1, j_mid], lon_2d[i_mid+1, j_mid]
        )
        
        # 2. Calculate dx (East-West distance) - Varies by Latitude (Row)
        # We calculate dx for every row along the central column
        lats_col = lat_2d[:, j_mid]
        lons_col = lon_2d[:, j_mid]
        lats_col_next = lat_2d[:, min(j_mid + 1, n_lons - 1)]
        lons_col_next = lon_2d[:, min(j_mid + 1, n_lons - 1)]
        
        dx_km_rows = _haversine_vec(lats_col, lons_col, lats_col_next, lons_col_next)
        
        # 3. Create 2D Map (dy is scalar, dx is vector)
        return np.tile(dx_km_rows[:, np.newaxis], (1, n_lons)) * dy_km

def prepare_grid_dict(ds):
    """
    Extracts coordinate arrays for area calculation.
    Updated to handle CORDEX 'rlat'/'rlon' and 2D 'latitude'/'longitude'.
    """
    lat_1d, lon_1d = None, None
    lat_2d, lon_2d = None, None

    # 1. Try to find 1D coordinates
    # We check dimension strictly == 1 to avoid picking up the 2D arrays here.
    for lat_name in ['lat', 'latitude', 'rlat']:
        if lat_name in ds.coords and ds[lat_name].ndim == 1:
            lat_1d = ds[lat_name].values
            break
            
    for lon_name in ['lon', 'longitude', 'rlon']:
        if lon_name in ds.coords and ds[lon_name].ndim == 1:
            lon_1d = ds[lon_name].values
            break
            
    # 2. Try to find 2D coordinates
    # Check 'lat2d' first, then 'latitude' (if 2D), then 'lat' (if 2D)
    if 'lat2d' in ds: 
        lat_2d = ds['lat2d'].values
    elif 'latitude' in ds and ds['latitude'].ndim == 2:
        lat_2d = ds['latitude'].values
    elif 'lat' in ds and ds['lat'].ndim == 2:
        lat_2d = ds['lat'].values

    if 'lon2d' in ds: 
        lon_2d = ds['lon2d'].values
    elif 'longitude' in ds and ds['longitude'].ndim == 2:
        lon_2d = ds['longitude'].values
    elif 'lon' in ds and ds['lon'].ndim == 2:
        lon_2d = ds['lon'].values

    # 3. Construct 2D mesh if we only found 1D
    # (This handles regular grids where no explicit 2D array is saved)
    if lat_2d is None and lat_1d is not None and lon_1d is not None:
        lat_2d, lon_2d = np.meshgrid(lat_1d, lon_1d, indexing='ij')

    if lat_2d is None:
        # Should now find 'latitude' and avoid this error
        raise ValueError(f"Could not determine grid coordinates from NetCDF. Found: {list(ds.coords)}")

    return {'lat_1d': lat_1d, 'lon_1d': lon_1d, 'lat2d': lat_2d, 'lon2d': lon_2d}


def calculate_precip_weighted_center(precip_data, mask, lat2d, lon2d):
    """Calculates the precipitation-weighted center of mass."""
    precip_in_mask = precip_data[mask]
    total_precip_weight = np.sum(precip_in_mask)

    if total_precip_weight == 0:
        return np.nan, np.nan

    lats_in_mask = lat2d[mask]
    lons_in_mask = lon2d[mask]

    weighted_lat_sum = np.sum(lats_in_mask * precip_in_mask)
    weighted_lon_sum = np.sum(lons_in_mask * precip_in_mask)

    center_lat = weighted_lat_sum / total_precip_weight
    center_lon = weighted_lon_sum / total_precip_weight

    return center_lat, center_lon

def calculate_kinematics(group):
    """
    Calculates kinematic properties of a track.

    Args:
        group (pd.DataFrame): DataFrame for a single track containing 'center_lat' and 'center_lon'.

    Returns:
        pd.Series: Contains 'total_distance_km', 'net_displacement_km', 'track_straightness'.
    """
    lats = group['center_lat'].values
    lons = group['center_lon'].values
    
    # Default for short tracks
    default_res = pd.Series([0.0, 0.0, 1.0], 
                            index=['total_distance_km', 'net_displacement_km', 'track_straightness'])

    if len(lats) < 2:
        return default_res

    # Filter NaNs
    valid_mask = np.isfinite(lats) & np.isfinite(lons)
    lats = lats[valid_mask]
    lons = lons[valid_mask]
    
    if len(lats) < 2:
        return default_res

    coords = list(zip(lats, lons))
    
    # Step-by-step distance sum
    total_dist = sum(great_circle(coords[i], coords[i+1]).km for i in range(len(coords)-1))
    
    # Net displacement (start to end)
    net_dist = great_circle(coords[0], coords[-1]).km
    
    # Straightness (1.0 = straight line, < 1.0 = meandering)
    straightness = net_dist / total_dist if total_dist > 0 else 1.0
    
    return pd.Series([total_dist, net_dist, straightness], 
                     index=['total_distance_km', 'net_displacement_km', 'track_straightness'])


def calculate_area_change(group):
    """
    Calculates area volatility (change between timesteps).

    Args:
        group (pd.DataFrame): DataFrame for a single track containing 'area_km2'.

    Returns:
        pd.Series: Contains 'max_area_volatility' and 'mean_area_volatility'.
    """
    areas = group['area_km2']
    
    if len(areas) < 2:
        return pd.Series([0.0, 0.0], index=['max_area_volatility', 'mean_area_volatility'])
        
    # Calculate absolute difference between consecutive steps
    diffs = areas.diff().abs()
    
    max_vol = diffs.max()
    mean_vol = diffs.mean()
    
    return pd.Series([max_vol, mean_vol], index=['max_area_volatility', 'mean_area_volatility'])