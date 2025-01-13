import xarray as xr
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import pandas as pd

# Load the NetCDF file
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

# Interpolate missing values
swh_filled = swh.interpolate_na(dim='time', method='linear')

# Reshape SWH data for EOF analysis
swh_np = swh_filled.values  # Shape: (time, lat, lon)
time_steps, lat_points, lon_points = swh_np.shape
swh_reshaped = swh_np.reshape(time_steps, -1)

# Perform PCA (EOF Analysis)
pca = PCA()
eofs = pca.fit_transform(swh_reshaped)
explained_variance = pca.explained_variance_ratio_ * 100

# Extract EOF2 and PC2
eof2 = pca.components_[1].reshape(lat_points, lon_points)
pc2 = eofs[:, 1]

# Plot EOF2 and PC2 together with the same color theme as EOF1 and PC1
fig, ax = plt.subplots(2, 1, figsize=(8, 10), gridspec_kw={"height_ratios": [3, 2], "hspace": 0.4})

# EOF2 (Spatial Pattern)
contour = ax[0].contourf(data['lon'], data['lat'], eof2, levels=20, cmap="coolwarm", extend='both')
cbar = fig.colorbar(contour, ax=ax[0], orientation="vertical", label="EOF2 Pattern")
ax[0].set_title("EOF2 Spatial Pattern", fontsize=14)
ax[0].set_xlabel("Longitude", fontsize=12)
ax[0].set_ylabel("Latitude", fontsize=12)

# PC2 (Temporal Variation)
ax[1].plot(time_years, pc2, label="PC2", color="blue", linewidth=1)
ax[1].set_title("Principal Component 2 (PC2)", fontsize=14)
ax[1].set_xlabel("Time (Years)", fontsize=12)
ax[1].set_ylabel("Amplitude", fontsize=12)
ax[1].grid()
ax[1].legend()

plt.tight_layout()
plt.savefig("EOF2_PC2_SameTheme_Report.png", dpi=300)  # Save plot for the report
plt.show()

