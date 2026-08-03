import xarray as xr
import numpy as np
import metpy.calc as mpcalc
from metpy.units import units
from pathlib import Path

# --- CONFIGURATION ---
# Pick a time where you know a storm existed but LI was positive
DATE_STR = "20000715" 
DATE = "2000-07"
HOUR_STR = "16" # 16:00 UTC

# Location: Central Alps (e.g., Tyrol/Swiss border)
TARGET_LAT = 47.0
TARGET_LON = 11.0

BASE_DIR = Path("/reloclim/dkn/data/ERA5")
# Adjust these paths to where your raw ERA5 input data lives
DIR_PL = BASE_DIR / "pressure_levels"  # T/q on 925, 850, 700, 500
DIR_SFC = BASE_DIR / "surface"         # 2t, 2d, sp, z (geopotential)

# --- PHYSICS ---
def calculate_li(T_parcel, Td_parcel, P_parcel, T_500, P_500=500*units.hPa):
    """Calculates Lifted Index for a specific parcel using MetPy (Bolton)."""
    try:
        # Calculate Parcel Profile
        parcel_profile = mpcalc.parcel_profile(
            [P_parcel, P_500], T_parcel, Td_parcel
        )
        T_lifted = parcel_profile[-1] # Temp at 500 hPa
        li = T_500 - T_lifted
        return li.m # Return magnitude
    except Exception as e:
        return np.nan

def main():
    print(f"--- DIAGNOSING STORM AT {TARGET_LAT}N, {TARGET_LON}E ---")
    
    # 1. Load Data (Simplified for single point)
    # You might need to adjust filename patterns

    # Load Pressure Levels (T, q)
    ds_pl = xr.open_mfdataset(f"{DIR_PL}/*{DATE}*.nc").sel(
        latitude=TARGET_LAT, longitude=TARGET_LON, method='nearest'
    ).sel(valid_time=f"{DATE_STR[:4]}-{DATE_STR[4:6]}-{DATE_STR[6:8]}T{HOUR_STR}:00")
    
    # Load Surface (2t, 2d, sp, z)
    ds_sfc = xr.open_mfdataset(f"{DIR_SFC}/*{DATE}*.nc").sel(
        latitude=TARGET_LAT, longitude=TARGET_LON, method='nearest'
    ).sel(valid_time=f"{DATE_STR[:4]}-{DATE_STR[4:6]}-{DATE_STR[6:8]}T{HOUR_STR}:00")

    # 2. Extract Variables
    # Surface
    T2m = ds_sfc['t2m'].values.item() * units.kelvin
    Td2m = ds_sfc['d2m'].values.item() * units.kelvin
    SP = ds_sfc['sp'].values.item() * units.pascal
    Z_sfc = ds_sfc['z'].values.item() / 9.80665 # Geopotential Height (m)

    # 500 hPa Env Temp
    T_500 = ds_pl['t'].sel(level=500).values.item() * units.kelvin
    
    print(f"\nConditions at {TARGET_LAT}, {TARGET_LON}:")
    print(f"  Terrain Height: {Z_sfc:.1f} m")
    print(f"  Surface Pressure: {SP.to('hPa'):.1f}")
    print(f"  500 hPa Temp:   {T_500.to('degC'):.1f}")
    
    print("\n--- PARCEL TESTS ---")
    
    # TEST 1: The Surface Parcel (The one you might be skipping)
    li_sfc = calculate_li(T2m, Td2m, SP, T_500)
    print(f"1. Surface Parcel (2m):")
    print(f"   Start: P={SP.to('hPa'):.1f}, T={T2m.to('degC'):.1f}, Td={Td2m.to('degC'):.1f}")
    print(f"   LI = {li_sfc:.2f} K  <-- {'UNSTABLE' if li_sfc < 0 else 'STABLE'}")

    # TEST 2: The Pressure Levels (The ones you are using)
    for lev in [925, 850, 700]:
        try:
            p_lev = lev * units.hPa
            t_lev = ds_pl['t'].sel(level=lev).values.item() * units.kelvin
            q_lev = ds_pl['q'].sel(level=lev).values.item() * units('kg/kg')
            
            # Calculate Dewpoint from q (approx)
            e = (q_lev * p_lev) / (0.622 + q_lev)
            td_lev = mpcalc.dewpoint(e)
            
            # Check if underground
            is_underground = p_lev > SP
            status_str = "[UNDERGROUND]" if is_underground else "[VALID]"
            
            li_lev = calculate_li(t_lev, td_lev, p_lev, T_500)
            
            print(f"\n{lev} hPa Parcel {status_str}:")
            print(f"   Start: T={t_lev.to('degC'):.1f}, Td={td_lev.to('degC'):.1f}")
            print(f"   LI = {li_lev:.2f} K")
            
        except Exception as e:
            print(f"\n{lev} hPa: Data missing or error ({e})")

if __name__ == "__main__":
    main()