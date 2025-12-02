import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy as np
import os
import pandas as pd
import matplotlib.colors as mcolors


def plot_precipitation(ax, lon, lat, prec, min_prec_threshold=0.1):
    """
    Plot the precipitation data on the map.

    Parameters:
    - ax: Matplotlib axes object.
    - lon: 2D array of longitudes.
    - lat: 2D array of latitudes.
    - prec: 2D array of precipitation values.
    - min_prec_threshold: Minimum precipitation threshold for masking (mm/hr).
    """

    # Set up the color scheme for precipitation
    cmap = plt.cm.RdYlGn_r  # Radar-like reversed colormap
    levels = [min_prec_threshold, 1, 2, 3, 4, 6, 8, 10, 15, 20]
    norm = plt.cm.colors.BoundaryNorm(levels, ncolors=cmap.N, clip=False)

    # Mask precipitation below the threshold
    prec_masked = np.ma.masked_less(prec, min_prec_threshold)

    # Make sure that data below the threshold is white or blue (ocean)
    cmap.set_under("white")
    cmap.set_bad("white")

    # Plot the precipitation data
    precip_plot = ax.pcolormesh(
        lon, lat, prec_masked, cmap=cmap, norm=norm, transform=ccrs.PlateCarree()
    )

    # Create a colorbar
    cbar = plt.colorbar(
        precip_plot, ax=ax, orientation="vertical", label="Precipitation (mm)"
    )

    return precip_plot


def plot_mcs_regions(ax, df_mcs):
    """
    Plot the identified MCS regions on the map.

    Parameters:
    - ax: Matplotlib axes object.
    - df_mcs: DataFrame containing MCS regions.
    """
    import numpy as np

    # Get unique region labels
    region_labels = df_mcs["region_label"].unique()

    # Define a colormap
    cmap = plt.cm.get_cmap("tab20", len(region_labels))

    # Plot each region with a different color
    for idx, region_label in enumerate(region_labels):
        region_data = df_mcs[df_mcs["region_label"] == region_label]
        ax.scatter(
            region_data["lon"],
            region_data["lat"],
            s=1,
            color=cmap(idx),
            transform=ccrs.PlateCarree(),
            label=f"Region {int(region_label)}",
        )

    # Optionally, add a legend
    # ax.legend(loc='upper right', markerscale=5, fontsize='small')


def save_detection_plot(
    lon, lat, prec, final_labeled_regions, file_time, output_dir, min_prec_threshold=0.1
):
    """
    Generate and save a plot of the detected MCS regions for a single time step.

    Parameters:
    - lon: 2D array of longitudes.
    - lat: 2D array of latitudes.
    - prec: 2D array of precipitation values.
    - final_labeled_regions: 2D array of labeled MCS regions.
    - file_time: The time associated with the file (used for the filename and title).
    - output_dir: Directory where the plot will be saved.
    - min_prec_threshold: Minimum precipitation threshold for masking (default is 0.1 mm).
    """

    # Create a DataFrame from the final_labeled_regions
    mcs_indices = np.where(final_labeled_regions > 0)
    if len(mcs_indices[0]) == 0:
        print(f"No MCS regions detected at time {file_time}. Skipping plot.")
        return

    mcs_lat = lat[mcs_indices]
    mcs_lon = lon[mcs_indices]
    mcs_prec_values = prec[mcs_indices]
    mcs_region_labels = final_labeled_regions[mcs_indices]

    df_mcs = pd.DataFrame(
        {
            "lat": mcs_lat,
            "lon": mcs_lon,
            "precip": mcs_prec_values,
            "region_label": mcs_region_labels,
        }
    )

    # Create a figure and axes with Cartopy projection
    fig, ax = plt.subplots(
        figsize=(12, 8), subplot_kw={"projection": ccrs.PlateCarree()}
    )

    # Set the extent of the map (adjust as necessary)
    ax.set_extent([lon.min(), lon.max(), lat.min(), lat.max()], crs=ccrs.PlateCarree())

    # Add map features
    ax.add_feature(cfeature.COASTLINE, linewidth=0.8)
    ax.add_feature(cfeature.BORDERS, linestyle=":", linewidth=0.5)
    ax.add_feature(cfeature.OCEAN, facecolor="lightblue")

    # Plot precipitation data
    plot_precipitation(ax, lon, lat, prec, min_prec_threshold)

    # Plot MCS regions
    plot_mcs_regions(ax, df_mcs)

    # Add title
    ax.set_title(f"MCS Detection at Time {file_time}")

    # Save the plot
    output_filename = f"mcs_detection_{file_time}.png"
    output_filepath = os.path.join(output_dir, output_filename)
    plt.savefig(output_filepath, dpi=300, bbox_inches="tight")
    plt.close(fig)  # Close the figure to free memory

    print(f"Saved plot for time {file_time} to {output_filepath}")


def save_intermediate_plots(detection_result, output_dir):
    """
    Generate and save intermediate plots for MCS detection, including:
    1. Precipitation data with moderate precipitation clusters overlaid (no legend or colorbar for clusters).
    2. Convective plumes.
    3. Final detected MCSs with precipitation data, colored according to MCS type with a legend.

    Parameters:
    - detection_result: Dictionary containing detection results.
    - output_dir: Directory where plots will be saved.
    """
    lon = detection_result["lon"]
    lat = detection_result["lat"]
    precipitation = detection_result["precipitation"]
    final_labeled_regions = detection_result["final_labeled_regions"]
    moderate_prec_clusters = detection_result["moderate_prec_clusters"]
    convective_plumes = detection_result["convective_plumes"]
    mcs_classification = detection_result["mcs_classification"]
    file_time = detection_result["time"]
    file_time_str = np.datetime_as_string(file_time, unit="h")

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # 1. Plot precipitation data with moderate precipitation clusters overlaid
    fig, ax = plt.subplots(
        figsize=(12, 8), subplot_kw={"projection": ccrs.PlateCarree()}
    )

    # Set the extent of the map
    ax.set_extent([lon.min(), lon.max(), lat.min(), lat.max()], crs=ccrs.PlateCarree())

    # Add map features
    ax.add_feature(cfeature.COASTLINE, linewidth=0.8)
    ax.add_feature(cfeature.BORDERS, linestyle=":", linewidth=0.5)
    ax.add_feature(cfeature.OCEAN, facecolor="lightblue")

    # Plot precipitation data
    plot_precipitation(ax, lon, lat, precipitation, min_prec_threshold=0.1)

    # Overlay moderate precipitation clusters (no legend or colorbar)
    clusters_mask = moderate_prec_clusters > 0
    ax.pcolormesh(
        lon,
        lat,
        np.ma.masked_where(~clusters_mask, moderate_prec_clusters),
        cmap="tab20",
        shading="auto",
        alpha=0.8,
        transform=ccrs.PlateCarree(),
    )

    # Add title
    ax.set_title(
        f"Precipitation with Moderate Precipitation Clusters at {file_time_str}"
    )

    # Save the plot
    plt.savefig(f"{output_dir}/moderate_clusters_{file_time_str}.png", dpi=300)
    plt.close(fig)

    # 2. Plot convective plumes
    fig, ax = plt.subplots(
        figsize=(12, 8), subplot_kw={"projection": ccrs.PlateCarree()}
    )

    # Set the extent of the map
    ax.set_extent([lon.min(), lon.max(), lat.min(), lat.max()], crs=ccrs.PlateCarree())

    # Add map features
    ax.add_feature(cfeature.COASTLINE, linewidth=0.8)
    ax.add_feature(cfeature.BORDERS, linestyle=":", linewidth=0.5)
    ax.add_feature(cfeature.OCEAN, facecolor="lightblue")

    # Plot convective plumes (no legend or colorbar)
    plumes_mask = convective_plumes > 0
    ax.pcolormesh(
        lon,
        lat,
        np.ma.masked_where(~plumes_mask, convective_plumes),
        cmap="tab20",
        shading="auto",
        alpha=0.5,
        transform=ccrs.PlateCarree(),
    )

    # Add title
    ax.set_title(f"Convective Plumes at {file_time_str}")

    # Save the plot
    plt.savefig(f"{output_dir}/convective_plumes_{file_time_str}.png", dpi=300)
    plt.close(fig)

    # 3. Plot final detected MCSs with precipitation data, colored according to MCS type with legend
    fig, ax = plt.subplots(
        figsize=(12, 10), subplot_kw={"projection": ccrs.PlateCarree()}
    )

    # Set the extent of the map
    ax.set_extent([lon.min(), lon.max(), lat.min(), lat.max()], crs=ccrs.PlateCarree())

    # Add map features
    ax.add_feature(cfeature.COASTLINE, linewidth=0.8)
    ax.add_feature(cfeature.BORDERS, linestyle=":", linewidth=0.5)
    ax.add_feature(cfeature.OCEAN, facecolor="lightblue")

    # Plot precipitation data
    plot_precipitation(ax, lon, lat, precipitation, min_prec_threshold=0.1)

    # Overlay MCS regions colored by MCS type
    # Create a mapping from MCS type to integer index
    mcs_types = sorted(list(set(mcs_classification.values())))
    type_to_index = {mcs_type: idx for idx, mcs_type in enumerate(mcs_types)}
    # Create a colormap
    cmap = plt.get_cmap("tab10", len(mcs_types))
    colors_list = [cmap(i) for i in range(len(mcs_types))]
    mcs_cmap = mcolors.ListedColormap(colors_list)

    # Create an array to hold the MCS type index for each grid point
    mcs_type_array = np.zeros(
        final_labeled_regions.shape, dtype=int
    )  # Default to zero (background)
    for label_value, mcs_type in mcs_classification.items():
        mask = final_labeled_regions == label_value
        mcs_type_array[mask] = (
            type_to_index[mcs_type] + 1
        )  # +1 to avoid zero (background)

    # Mask the background (zero) values
    mcs_type_array_masked = np.ma.masked_where(mcs_type_array == 0, mcs_type_array)

    # Create normalization
    norm = mcolors.Normalize(vmin=1, vmax=len(mcs_types))

    # Overlay the MCS regions
    ax.pcolormesh(
        lon,
        lat,
        mcs_type_array_masked,
        cmap=mcs_cmap,
        norm=norm,
        shading="auto",
        transform=ccrs.PlateCarree(),
    )

    # Add title
    ax.set_title(f"MCS Types at {file_time_str}")

    # Create custom legend
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=colors_list[idx])
        for idx in range(len(mcs_types))
    ]
    if handles:
        ax.legend(
            handles,
            mcs_types,
            title="MCS Types",
            loc="lower center",
            bbox_to_anchor=(0.5, -0.15),
            ncol=len(mcs_types),
        )

        # Adjust layout to make room for the legend
        plt.subplots_adjust(bottom=0.25)
        # Save the plot
        plt.savefig(
            f"{output_dir}/mcs_types_{file_time_str}.png", dpi=300, bbox_inches="tight"
        )

    plt.close(fig)
