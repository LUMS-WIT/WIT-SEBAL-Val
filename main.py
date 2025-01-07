"""
Created on Jan 2025

validation of sebal soil moisture estimates
using WITSMS Network

@author: hamza rafique
"""
import os
from utils import SoilMoistureData, SebalSoilMoistureData
from utils import remove_nan_entries, save_to_excel, save_metadata, save_to_plot
from scaling import scaling, temporal_matching
from sms_calibration import sms_calibrations

from config import ROW_PATH, WIT_SMS_PATH, RASTER_FOLDER_PATH, TEMPORAL_WIN, \
      VALIDATION_FOLDER, RESCALING, SAVE_PLOT, IMAGES_FOLDER, METADATA_FILE_PATH

if not os.path.exists(VALIDATION_FOLDER):
    # Create the folder
    os.makedirs(VALIDATION_FOLDER)

# ----------------------------------------------------------------------
# STEP 1 : Reading sms and raster data, finding the overlapping points 
# and saving them in excel files along with metadata file
# ----------------------------------------------------------------------

soil_moisture_data = SoilMoistureData(WIT_SMS_PATH)
soil_moisture_data.read_data()
metadata = soil_moisture_data.get_metadata()
raster_data = SebalSoilMoistureData(RASTER_FOLDER_PATH)


validation_metadata = []
print(f'Performing temporal matching with temporal window of {TEMPORAL_WIN} days')
# for _metadata in data_within_extent:
for _metadata in metadata:
    lat, lon = float(_metadata['latitude']), float(_metadata['longitude'])


    csv_dates, csv_values = soil_moisture_data.get_soil_moisture_by_location(lat, lon)
    raster_dates, raster_values = raster_data.get_data(lat, lon)

    # Check if None returned (No overalping points), then skip to the next iteration
    if raster_dates is None or raster_values is None:
        continue  # Skip the rest of the code in the loop and move to the next iteration

    csv_dates, csv_values = remove_nan_entries(csv_dates, csv_values)
    raster_dates, raster_values = remove_nan_entries(raster_dates, raster_values)

    # Find overlapping data
    ref_data = (csv_dates, csv_values)  # Reference data from CSV files
    test_data = (raster_dates, raster_values)  # Test data from raster files
    ref_data = sms_calibrations(ref_data)

    if RESCALING:
        test_data, _ = scaling(test_data, ref_data)
        csv_dates, csv_values = ref_data
        raster_dates, raster_values = test_data
    # Find overlapping data
    common_dates, ref_values, test_values, overlap_count = temporal_matching(test_data, ref_data, TEMPORAL_WIN)
    # print(raster_values)
    # print("Overlapping Dates:", common_dates)
    # print("Reference Values:", ref_values)
    # print("Test Values:", test_values)
    # print("Total Valid Overlaps:", overlap_count)


    _metadata['overlaps'] = overlap_count

    # Only add to the new list if overlaps are non-zero
    if overlap_count > 0:
        validation_metadata.append(_metadata)

        # Prepare data to save in Excel
        data_to_save = list(zip(common_dates, ref_values, test_values))
        headers = ['Timestamp', 'wit_sm', 'sebal_sm']
        filename = f'\sebal_{ROW_PATH}_witgpi_{_metadata["gpi"]}_lat_{lat}_lon_{lon}.xlsx'

        filename = VALIDATION_FOLDER + filename

        # Save data to Excel file
        save_to_excel(data_to_save, filename, headers)

        filename_img = f'\sebal_{ROW_PATH}_witgpi_{_metadata["gpi"]}_lat_{lat}_lon_{lon}.png'
        filename_img = IMAGES_FOLDER + filename_img
        if SAVE_PLOT:
            save_to_plot(csv_dates, csv_values, raster_dates, raster_values, lat, lon, filename_img)

print('Total number of overlapping in-situ sensors', len(validation_metadata))

if not RESCALING:
    save_metadata(validation_metadata, METADATA_FILE_PATH)