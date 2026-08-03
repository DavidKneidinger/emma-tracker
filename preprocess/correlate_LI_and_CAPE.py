import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import warnings
from scipy.stats import binned_statistic

# Ignore serialization warnings
warnings.filterwarnings("ignore")

# --- CONFIGURATION ---

# 1. Paths
LI_DIR = Path("/reloclim/dkn/data/ERA5/lifted_index_corr")
CAPE_DIR = Path("/reloclim/dkn/data/ERA5/surface") 

# 2. Date to Check
YEAR = 2000
MONTH = '07'

# 3. Output
OUTPUT_PLOT = f"cape_vs_li_validation_{YEAR}{MONTH}.png"

# --- PHYSICS CURVE ---
def pucik_curve(li):
    """Pucik et al. (2017) polynomial fit: y = 85.2 - 167.9*x + 17.9*x^2"""
    return 85.2284 - (167.9808 * li) + (17.9118 * li**2)

def main():
    print(f"--- Running Validation for {YEAR}-{MONTH} ---")
    
    li_path = LI_DIR / str(YEAR) / MONTH
    
    # --- 1. LOAD LI DATA (Manual Step-by-Step) ---
    print("Loading LI data...")
    datasets = []
    li_files = sorted(list(li_path.glob("*.nc")))
    
    if not li_files:
        print(f"❌ No LI files found in {li_path}")
        return

    for f in li_files:
        try:
            ds = xr.open_dataset(f)
            
            # Promote valid_time to dimension
            if 'valid_time' in ds.coords:
                t_val = ds['valid_time'].values
                if np.ndim(t_val) == 0: t_val = [t_val]
                ds = ds.expand_dims(time=t_val)
                if 'valid_time' in ds.coords and 'valid_time' not in ds.dims:
                    ds = ds.drop_vars('valid_time')
                    
            elif 'time' in ds.coords:
                if 'time' not in ds.dims:
                    ds = ds.expand_dims(time=ds['time'].values)

            datasets.append(ds)
        except Exception as e:
            print(f"Skipping {f.name}: {e}")
    
    if not datasets:
        print("No datasets loaded.")
        return

    print(f"Concatenating {len(datasets)} files...")
    ds_li = xr.concat(datasets, dim='time')

    # --- 2. LOAD CAPE DATA ---
    print("Loading CAPE data...")
    cape_file_pattern = f"{CAPE_DIR}/*{YEAR}*{MONTH}*_CAPE.nc"
    try:
        ds_cape = xr.open_mfdataset(cape_file_pattern, combine='by_coords')
    except OSError:
        # Fallback to SP if CAPE is stored there
        ds_cape = xr.open_mfdataset(f"{CAPE_DIR}/*{YEAR}*{MONTH}*_SP.nc", combine='by_coords')

    # Standardize CAPE names
    if 'valid_time' in ds_cape.coords:
        ds_cape = ds_cape.rename({'valid_time': 'time'})
    if 'longitude' in ds_cape.coords:
        ds_cape = ds_cape.rename({'longitude': 'lon', 'latitude': 'lat'})
    
    ds_cape = ds_cape.drop_vars(['expver', 'number'], errors='ignore')

    # --- 3. EXTRACT VARS ---
    li_var = 'LI'
    cape_var = None
    for v in ['cape', 'convective_available_potential_energy', 'var59']:
        if v in ds_cape:
            cape_var = v
            break
            
    if not cape_var:
        print("❌ CAPE variable not found.")
        return

    # --- 4. ALIGN & FLATTEN ---
    print("Aligning and flattening...")
    common_times = np.intersect1d(ds_li.time.values, ds_cape.time.values)
    print(f"Found {len(common_times)} matching hours.")
    
    # Subsample for speed/memory (every 5th hour)
    subset = common_times[::5]
    
    x_li = ds_li[li_var].sel(time=subset).values.flatten()
    y_cape = ds_cape[cape_var].sel(time=subset).values.flatten()
    
    # Basic Validity Mask (Finite values)
    mask_valid = np.isfinite(x_li) & np.isfinite(y_cape)
    x_li = x_li[mask_valid]
    y_cape = y_cape[mask_valid]

    # --- 5. ENHANCED PLOTTING ---
    print("Calculating statistics and plotting...")
    
    # A. Filter for the "Active" Region
    # We hide the massive pile of stable points (CAPE~0) to focus on the correlation
    mask_active = (x_li > -16) & (x_li < 3) & (y_cape > 50) & (y_cape < 6000)
    x_plot = x_li[mask_active]
    y_plot = y_cape[mask_active]
    
    # B. Binned Statistics (The "Skeleton" of your data)
    # This draws a line through the median CAPE for every 0.5 K of LI
    bin_edges = np.linspace(-15, 2, 35)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    median_cape, _, _ = binned_statistic(x_plot, y_plot, statistic='median', bins=bin_edges)
    p25_cape, _, _ = binned_statistic(x_plot, y_plot, statistic=lambda y: np.percentile(y, 25), bins=bin_edges)
    p75_cape, _, _ = binned_statistic(x_plot, y_plot, statistic=lambda y: np.percentile(y, 75), bins=bin_edges)

    # C. Plot
    plt.figure(figsize=(10, 8))
    
    # 1. Hexbin (Raw Data Density)
    plt.hexbin(x_plot, y_plot, gridsize=80, cmap='gist_yarg', bins='log', mincnt=1, alpha=0.5, label='Data Density')
    cb = plt.colorbar()
    cb.set_label('Log Count (Grid Points)')
    
    # 2. Pucik Curve (Theory)
    x_curve = np.linspace(-15, 2, 100)
    y_curve = pucik_curve(x_curve)
    plt.plot(x_curve, y_curve, color='cyan', linestyle='--', linewidth=3, label="Púčik et al. (2017) Fit")
    
    # 3. Your Data (Median + IQR)
    plt.plot(bin_centers, median_cape, color='red', linewidth=2, marker='o', markersize=4, label="Your Data (Median)")
    plt.fill_between(bin_centers, p25_cape, p75_cape, color='red', alpha=0.15, label="Your Data (IQR)")
    
    # Formatting
    plt.xlim(2, -14) # Reversed axis (Unstable left)
    plt.ylim(0, 4000)
    plt.xlabel("Lifted Index (K)")
    plt.ylabel("ERA5 CAPE (J/kg)")
    plt.title(f"Physical Validation: {YEAR}-{MONTH}\nComparison of New Calculation vs. Púčik et al. (2017)")
    plt.legend(loc='upper right', frameon=True)
    plt.grid(True, alpha=0.3)
    
    plt.savefig(OUTPUT_PLOT, dpi=200)
    print(f"✅ Plot saved to: {OUTPUT_PLOT}")

if __name__ == "__main__":
    main()