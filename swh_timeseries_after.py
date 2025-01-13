import xarray as xr
import pandas as pd
import numpy as np

# Load the NetCDF file without decoding times initially
file_path = "preprocessed_waveht_cleaned.nc"
data = xr.open_dataset(file_path, decode_times=False)

# Inspect the 'time' variable
print("Inspecting 'time' variable:")
print(data['time'])
print(data['time'].attrs)
print(data['time'].values[:10])

# Decode the 'time' variable manually
try:
    raw_time = data['time'].values
    # Extract the integer and fractional parts
    date_part = raw_time.astype(int).astype(str)  # Extract YYYYMMDD as strings
    frac_part = raw_time - raw_time.astype(int)  # Extract fractional part (fraction of the day)

    # Convert date_part to datetime and add the fractional part as timedelta
    time_decoded = pd.to_datetime(date_part, format='%Y%m%d') + pd.to_timedelta(frac_part * 24, unit='h')
    data['time'] = ('time', time_decoded)  # Assign back to the dataset
except Exception as e:
    print(f"Error decoding time: {e}")

# Remove duplicate time entries
print("Checking for duplicate time values...")
duplicates = data['time'].to_series().duplicated().sum()
print(f"Number of duplicates in 'time': {duplicates}")

if duplicates > 0:
    # Keep only unique time values
    is_unique = ~data['time'].to_series().duplicated().values
    data = data.isel(time=is_unique)

# Sort and ensure unique time index
data = data.sortby('time')

# Convert time to fractional years
time = data['time']
time_years = time.dt.year + (time.dt.dayofyear - 1) / 365.25

# Extract Significant Wave Height (SWH)
try:
    swh = data['var229']  # Replace 'var229' with your actual SWH variable name
except KeyError:
    raise KeyError("Variable 'var229' not found in the dataset. Verify the variable name.")

# Check for missing data
missing_data_fraction = swh.isnull().sum(dim=('lat', 'lon')) / (len(data['lat']) * len(data['lon']))
print("Missing data fraction over time:")
print(missing_data_fraction)

# Interpolate missing data
swh_filled = swh.interpolate_na(dim='time', method='linear')

# Save the preprocessed data into a new NetCDF file
output_file = "preprocessed_waveht_final.nc"
preprocessed_data = xr.Dataset({
    'swh': (('time', 'lat', 'lon'), swh_filled.values),
}, coords={
    'time': time,
    'lat': data['lat'],
    'lon': data['lon'],
})

# Save using netCDF4 engine
preprocessed_data.to_netcdf(output_file, format='NETCDF4', engine='netcdf4')
print(f"Preprocessed data saved to {output_file}")

