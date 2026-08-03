import xarray as xr
from pathlib import Path

# --- CONFIGURATION ---
INPUT_BASE = Path("/reloclim/dkn/data/ERA5/lifted_index_remap")
OUTPUT_BASE = Path("/reloclim/dkn/data/ERA5/lifted_index_remap_yearly")

# Set the year range (inclusive)
YEARS = range(2025, 2026) 
TARGET_MONTHS = ['05', '06', '07', '08', '09']

# Ensure output directory exists
OUTPUT_BASE.mkdir(parents=True, exist_ok=True)

def aggregate_year(year):
    print(f"\n--- Starting Processing for Year {year} ---")
    
    # 1. Gather all file paths for MJJAS
    file_paths = []
    for month in TARGET_MONTHS:
        month_dir = INPUT_BASE / str(year) / month
        if month_dir.exists():
            # Grab all lifted_index hourly files and sort them
            files = sorted(month_dir.glob("lifted_index_*.nc"))
            file_paths.extend(files)
        else:
            print(f"   ⚠️ Directory not found: {month_dir}")
            
    if not file_paths:
        print(f"   ❌ No files found for {year}. Skipping.")
        return

    print(f"   Found {len(file_paths)} hourly files. Loading metadata...")

    # 2. Open all files as a single virtual dataset
    # The kwargs here are specifically chosen to speed up loading thousands of files
    # by preventing xarray from strictly comparing identical spatial coordinates in every file.
    ds = xr.open_mfdataset(
        file_paths,
        combine='nested',
        concat_dim='time',
        coords='minimal',       
        compat='override',      
        data_vars='minimal',
        parallel=False          # Uses dask to read files concurrently
    )
    
    # Ensure strict chronological order
    ds = ds.sortby('time')

    # 3. Pull everything into RAM
    # Since you have 150+ GB of RAM and this is ~1.5 GB, we load it entirely into memory.
    # This prevents the disk from seeking back and forth during the save process.
    print(f"   Pulling {year} data into RAM (this may take a minute)...")
    ds.load()

    # 4. Save to a single compressed NetCDF file
    output_file = OUTPUT_BASE / f"li_mjjas_{year}.nc"
    print(f"   Saving concatenated file to {output_file.name}...")
    
    # Apply standard NetCDF4 compression to match ERA5 sizes
    encoding = {var: {'zlib': True, 'complevel': 4} for var in ds.data_vars}
    
    ds.to_netcdf(output_file, encoding=encoding)
    
    # Free up RAM
    ds.close()
    print(f"   ✅ {year} completed successfully!")

if __name__ == "__main__":
    print(f"📁 Output Directory: {OUTPUT_BASE}")
    for target_year in YEARS:
        aggregate_year(target_year)
    
    print("\n🎉 All requested years have been aggregated!")