import xarray as xr

# Load the NetCDF file
file_path = "preprocessed_waveht_cleaned.nc"
data = xr.open_dataset(file_path)

# Check for NaN values
swh = data['var229']
nan_count = swh.isnull().sum().item()
print(f"Number of NaN values in the dataset: {nan_count}")

