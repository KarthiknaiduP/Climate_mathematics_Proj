import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# Load the NetCDF file using xarray
file_path = "preprocessed_waveht_cleaned.nc"
data = xr.open_dataset(file_path)

# Extract variables
lon = data['lon']
lat = data['lat']
swh = data['var229']  # Significant Wave Height

# Calculate climatological mean (mean SWH over time)
swh_mean = swh.mean(dim='time')

# Plot the spatial pattern of mean SWH with cartopy for geographic context
fig = plt.figure(figsize=(10, 6))
ax = plt.axes(projection=ccrs.PlateCarree())
swh_mean.plot(
    ax=ax,
    cmap="coolwarm",  # Updated color theme for better distinction
    cbar_kwargs={
        'label': 'Mean SWH (m)',
        'shrink': 0.8,  # Adjust the colorbar size
        'pad': 0.1      # Add space between colorbar and plot
    },
    levels=20,  # Moderate granularity for better readability
    extend='both'  # Add arrows to colorbar for extreme values
)

# Add coastlines for geographic context
ax.coastlines(resolution='50m', color='black', linewidth=0.8)

# Add latitude and longitude labels manually
ax.set_xticks(range(30, 121, 10), crs=ccrs.PlateCarree())  # Longitude ticks
ax.set_yticks(range(-60, 31, 10), crs=ccrs.PlateCarree())  # Latitude ticks
ax.set_xlabel("Longitude", fontsize=12, labelpad=10)
ax.set_ylabel("Latitude", fontsize=12, labelpad=10)
ax.tick_params(labelsize=10)  # Adjust tick label size for better visibility

# Set titles and labels
plt.title("Climatological Mean Significant Wave Height (1993-2023)", fontsize=14, pad=10)

# Adjust layout and save the plot
plt.tight_layout()
plt.savefig("Climatological_Mean_SWH_Final_Labeled.png", dpi=300)  # Save the updated plot
plt.show()

