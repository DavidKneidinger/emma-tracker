import xarray as xr
import numpy as np
from pathlib import Path

# Adjust paths to one of your actual files
PL_FILE = "/reloclim/dkn/data/ERA5/pressure_level/2000-06_LI.nc"
SP_FILE = "/reloclim/dkn/data/ERA5/surface/2000-06_SP.nc"

def check_data():
    print("--- DIAGNOSTIC CHECK ---")
    
    # 1. LOAD
    ds_pl = xr.open_dataset(PL_FILE)
    ds_sp = xr.open_dataset(SP_FILE)
    
    # 2. CHECK TIME ALIGNMENT
    t_pl = ds_pl.valid_time.values
    t_sp = ds_sp.valid_time.values
    
    if not np.array_equal(t_pl, t_sp):
        print("CRITICAL WARNING: Time coordinates do NOT match!")
        print(f"PL times: {len(t_pl)}, SP times: {len(t_sp)}")
        print(f"First PL: {t_pl[0]}, First SP: {t_sp[0]}")
    else:
        print("✅ Time coordinates are perfectly aligned.")

    # 3. CHECK MASKING IN ALPS
    # Define an Alpine box (approx)
    lat_min, lat_max = 45.0, 48.0
    lon_min, lon_max = 6.0, 13.0
    
    sp_alps = ds_sp['sp'].sel(latitude=slice(lat_max, lat_min), longitude=slice(lon_min, lon_max)).isel(valid_time=12) # Noon
    
    # Check 850 hPa
    p_850 = 85000.0
    mask_850 = p_850 < sp_alps
    
    valid_fraction = mask_850.sum() / mask_850.size
    print(f"\n--- 850 hPa Validity in Alps (12:00 UTC) ---")
    print(f"Percentage of Alps pixels where 850 hPa is valid (above ground): {valid_fraction.values*100:.1f}%")
    
    if valid_fraction < 0.1:
        print("-> CONCLUSION: 850 hPa is underground for >90% of the Alps.")
        print("   Your code is using ONLY 700 hPa for these pixels.")

if __name__ == "__main__":
    check_data()