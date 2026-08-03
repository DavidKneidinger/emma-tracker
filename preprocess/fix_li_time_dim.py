import xarray as xr
from pathlib import Path

# --- CONFIGURATION ---
LI_FILE = "/reloclim/dkn/data/ERA5/lifted_index_remap_yearly/li_mjjas_YYYY.nc"
PRECIP_FILE = "/reloclim/dkn/data/IMERG/imerg_accum/IMERG_Hourly_Accumulation_May-Sep_YYYY.nc"
OUTPUT_FILE = "/reloclim/dkn/data/ERA5/lifted_index_remap_yearly/li_mjjas_YYYY_TIMEFIXED.nc"

years = range(2025, 2026)  

def main():
    print("⏳ Loading datasets...")
    for year in years:

        li_file = LI_FILE.replace("YYYY", str(year))
        precip_file = PRECIP_FILE.replace("YYYY", str(year))
        output_file = OUTPUT_FILE.replace("YYYY", str(year))

        if Path(output_file).exists():
            print(f"⚠️  Output file for {year} already exists. Skipping...")
            continue

        ds_l = xr.open_dataset(li_file)
        ds_p = xr.open_dataset(precip_file)

        print(f"   LI Time length:     {len(ds_l.time)}")
        print(f"   Precip Time length: {len(ds_p.time)}")

        if len(ds_l.time) != len(ds_p.time):
            print("❌ Error: Time lengths do not match! Cannot copy time array.")
            return

        print("\n🔧 Injecting identical time coordinates from Precip into LI...")
        # .values extracts the raw datetime64 array, bypassing any coordinate conflicts
        ds_l['time'] = ds_p.time.values

        # Clean up the metadata so it doesn't say "days" anymore
        ds_l['time'].attrs = ds_p['time'].attrs

        print(f"💾 Saving fixed file to: {output_file}")
        # Re-apply compression
        encoding = {var: {'zlib': True, 'complevel': 4} for var in ds_l.data_vars}
        ds_l.to_netcdf(output_file, encoding=encoding)
        
        ds_l.close()
        ds_p.close()
        print(f"\n✅ Done! Point your tracker to {output_file}")

if __name__ == "__main__":
    main()