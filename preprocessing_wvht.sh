#!/bin/bash

# Input and output files
INPUT_FILE="merged_waveht.nc"
OUTPUT_FILE="preprocessed_waveht_final.nc"

# Step 1: Sort or directly process the file
echo "Selecting valid time steps..."
if cdo seltime,1993-01-01T00:00:00,2023-12-31T23:00:00 $INPUT_FILE sorted_waveht.nc; then
    echo "Time steps selected successfully."
else
    echo "Skipping time selection. Using the entire file instead."
    cp $INPUT_FILE sorted_waveht.nc
fi

# Step 2: Fill missing values with linear interpolation
echo "Interpolating missing values..."
cdo inttime,1993-01-01,00:00:00,1hour sorted_waveht.nc interpolated_waveht.nc

# Step 3: Set remaining missing values to zero
echo "Setting remaining missing values to 0..."
cdo setmisstoc,0 interpolated_waveht.nc filled_waveht.nc

# Step 4: Crop to the Indian Ocean region
echo "Cropping to the Indian Ocean region..."
cdo sellonlatbox,30,120,-60,30 filled_waveht.nc cropped_waveht.nc

# Step 5: Aggregate to daily means (optional)
echo "Averaging hourly data to daily means..."
cdo daymean cropped_waveht.nc daily_waveht.nc

# Step 6: Save the final output file
echo "Saving the final cleaned file..."
mv daily_waveht.nc $OUTPUT_FILE

# Cleanup intermediate files
echo "Cleaning up intermediate files..."
rm sorted_waveht.nc interpolated_waveht.nc filled_waveht.nc cropped_waveht.nc

# Verify the final output
echo "Verifying the final output file..."
cdo sinfo $OUTPUT_FILE

echo "Preprocessing complete. Output file: $OUTPUT_FILE"

