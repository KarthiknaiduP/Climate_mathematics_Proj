import xarray as xr
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import pandas as pd

# Load the NetCDF file using xarray
file_path = "preprocessed_waveht_cleaned.nc"
data = xr.open_dataset(file_path, decode_times=False)

# Decode the 'time' variable manually
raw_time = data['time'].values
date_part = raw_time.astype(int).astype(str)  # Extract YYYYMMDD as strings
frac_part = raw_time - raw_time.astype(int)  # Extract fractional part (fraction of the day)
time_decoded = pd.to_datetime(date_part, format='%Y%m%d') + pd.to_timedelta(frac_part * 24, unit='h')
data['time'] = ('time', time_decoded)  # Assign back to the dataset

# Remove duplicate time entries
is_unique = ~data['time'].to_series().duplicated().values
data = data.isel(time=is_unique)

# Sort and ensure unique time index
data = data.sortby('time')

# Convert time to fractional years
time = data['time']
time_years = time.dt.year + (time.dt.dayofyear - 1) / 365.25

# Extract Significant Wave Height (SWH)
swh = data['var229']  # Replace 'var229' with your actual SWH variable name

# Check for missing data and interpolate
swh_filled = swh.interpolate_na(dim='time', method='linear')

# Reshape SWH data for EOF analysis
swh_np = swh_filled.values  # Shape: (time, lat, lon)
time_steps, lat_points, lon_points = swh_np.shape
swh_reshaped = swh_np.reshape(time_steps, -1)

# Perform PCA (EOF Analysis)
pca = PCA()
eofs = pca.fit_transform(swh_reshaped)
explained_variance = pca.explained_variance_ratio_ * 100

# Cumulative variance explained
cumulative_variance = np.cumsum(explained_variance)

# Plot percentage variance (individual EOFs) and cumulative variance
fig, ax1 = plt.subplots(figsize=(8, 5))

# Plot percentage variance for the first 15 EOFs
ax1.plot(range(1, 16), explained_variance[:15], marker='o', color='black', label='Percentage Variance')
ax1.set_xlabel('Mode Number', fontsize=12)
ax1.set_ylabel('Percentage Variance [%]', fontsize=12, color='black')
ax1.tick_params(axis='y', labelcolor='black')
ax1.grid()
ax1.legend(loc='center right', bbox_to_anchor=(0.85, 0.5))  # Adjusted legend position to avoid overlap

# Plot cumulative variance (on a second y-axis)
ax2 = ax1.twinx()
ax2.plot(range(1, 16), cumulative_variance[:15], marker='o', color='blue', label='Cumulative Variance')
ax2.set_ylabel('Cumulative Variance [%]', fontsize=12, color='blue')
ax2.tick_params(axis='y', labelcolor='blue')
ax2.legend(loc='center right', bbox_to_anchor=(0.85, 0.4))  # Adjusted legend position to avoid overlap

# Set x-axis ticks to display all numbers from 1 to 15
ax1.set_xticks(range(1, 16))

# Set the title
plt.title('Eigenvalues of Covariance Matrix', fontsize=14)

# Save and show the plot
plt.tight_layout()
plt.savefig("Cumulative_Variance_15EOFs_Updated_Report.png", dpi=300)  # Save for the report
plt.show()

