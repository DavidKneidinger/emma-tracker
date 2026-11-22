import xarray as xr
import os
import sys


def process_netcdf_file(source_path, dest_path):
    """
    Reads a NetCDF file, modifies its attributes and variables,
    and saves it to a new location based on specific requirements.

    Args:
        source_path (str): The full path to the source NetCDF file.
        dest_path (str): The full path where the modified NetCDF file will be saved.
    """
    try:
        # Open the dataset using a 'with' statement to ensure it's properly closed.
        with xr.open_dataset(source_path) as ds:
            # --- Modification 1: Change global attribute 'Units' to 'units' ---
            if "Units" in ds.attrs:
                ds.attrs["units"] = ds.attrs.pop("Units")

            # --- Modification 2: Update attributes for the 'LI' variable ---
            # Check if the 'LI' variable exists before modifying it.
            if "LI" in ds.variables:
                # Set the units attribute specifically to 'K'.
                ds["LI"].attrs["units"] = "K"
                # Remove the 'coordinates' attribute as 'expver' and 'number' will be dropped.
                if "coordinates" in ds["LI"].attrs:
                    del ds["LI"].attrs["coordinates"]

            # --- Modification 3: Drop the 'expver' and 'number' coordinates/variables ---
            # This will remove them if they exist as coordinates or data variables.
            # The 'errors="ignore"' argument prevents an error if they don't exist in a file.
            vars_to_drop = ["expver", "number"]
            # ds_modified = ds.drop_vars(vars_to_drop, errors='ignore')
            ds_modified = ds.drop(["number", "expver"])
            # --- Saving the file ---
            # Get the directory part of the destination path.
            dest_folder = os.path.dirname(dest_path)
            # Create the destination directory if it doesn't already exist.
            # exist_ok=True prevents an error if the folder is already there.
            os.makedirs(dest_folder, exist_ok=True)

            # Save the modified dataset to the new file path.
            ds_modified.to_netcdf(dest_path)
            # print(f"Successfully processed: {source_path} -> {dest_path}")

    except Exception as e:
        # If any error occurs during the process, print an error message.
        print(f"ERROR processing file {source_path}: {e}", file=sys.stderr)


def main():
    """
    Main function to traverse the directory structure and process all NetCDF files.
    """
    # --- Configuration ---
    # Define the source and destination root directories.
    source_dir = "/reloclim/dkn/data/ERA5/lifting_index_old"
    dest_dir = "/reloclim/dkn/data/ERA5/lifting_index"

    print(f"Starting NetCDF file processing.")
    print(f"Source: {source_dir}")
    print(f"Destination: {dest_dir}")
    print("-" * 30)

    # --- Directory Traversal ---
    # os.walk efficiently traverses the directory tree (root, subdirectories, files).
    for root, dirs, files in os.walk(source_dir):
        for filename in files:
            # Process only files that end with the .nc extension.
            if filename.endswith(".nc"):
                # Construct the full path to the source file.
                source_path = os.path.join(root, filename)

                # Determine the relative path of the file from the source directory.
                # This preserves the YYYY/MM structure.
                relative_path = os.path.relpath(source_path, source_dir)

                # Construct the full destination path for the new file.
                dest_path = os.path.join(dest_dir, relative_path)

                # Call the function to process the individual file.
                process_netcdf_file(source_path, dest_path)

    print("-" * 30)
    print("Processing complete.")


if __name__ == "__main__":
    # This ensures the main() function is called only when the script is executed directly.
    main()
