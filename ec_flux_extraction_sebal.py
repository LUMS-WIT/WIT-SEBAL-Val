"""
1.

Load AmeriFlux daily CSV, clean it, and subset to a date range and desired variables. 
This data is meant for SEBAL (meteo)

Returns
-------
pd.DataFrame
    Cleaned dataframe indexed by DATE with the selected variables.

Raises
------
ValueError
    If none of the expected key variables are present in the CSV.
"""

import tarfile
import re
import os
import pandas as pd
import numpy as np

input_root = r"D:\SEBAL\datasets\landsat\calibrations"
output_root = r"D:\SEBAL\datasets\SEBAL_out\calibrations"


flux_daily_path = r"D:\SEBAL\datasets\EC Flux\Ameriflux\BsH\AMF_US-Rls_FLUXNET_SUBSET_2014-2023_5-7\AMF_US-Rls_FLUXNET_SUBSET_DD_2014-2023_5-7.csv"
flux_halfhourly_path = r"D:\SEBAL\datasets\EC Flux\Ameriflux\BsH\AMF_US-Rls_FLUXNET_SUBSET_2014-2023_5-7\AMF_US-Rls_FLUXNET_SUBSET_HH_2014-2023_5-7.csv"
out_csv = "landsat_meteo_match_RIs.csv"

# flux_daily_path = r"D:\SEBAL\datasets\EC Flux\Ameriflux\BsH\AMF_US-Rms_FLUXNET_SUBSET_2014-2023_5-7\AMF_US-Rms_FLUXNET_SUBSET_DD_2014-2023_5-7.csv"
# flux_halfhourly_path = r"D:\SEBAL\datasets\EC Flux\Ameriflux\BsH\AMF_US-Rms_FLUXNET_SUBSET_2014-2023_5-7\AMF_US-Rms_FLUXNET_SUBSET_HH_2014-2023_5-7.csv"
# out_csv = "landsat_meteo_match_Rms.csv"

# flux_daily_path = r"D:\SEBAL\datasets\EC Flux\Ameriflux\BsH\AMF_US-Rwf_FLUXNET_SUBSET_2014-2023_5-7\AMF_US-Rwf_FLUXNET_SUBSET_DD_2014-2023_5-7.csv"
# flux_halfhourly_path = r"D:\SEBAL\datasets\EC Flux\Ameriflux\BsH\AMF_US-Rwf_FLUXNET_SUBSET_2014-2023_5-7\AMF_US-Rwf_FLUXNET_SUBSET_HH_2014-2023_5-7.csv"
# out_csv = "landsat_meteo_match_Rwf.csv"


# input_root = r"D:\SEBAL\datasets\landsat\calibrations\Bwh"

# flux_daily_path = r"D:\SEBAL\datasets\EC Flux\Ameriflux\Bwh\AMF_US-SRC_FLUXNET_SUBSET_2008-2014_5-7\AMF_US-SRC_FLUXNET_SUBSET_DD_2008-2014_5-7.csv"
# flux_halfhourly_path = r"D:\SEBAL\datasets\EC Flux\Ameriflux\Bwh\AMF_US-SRC_FLUXNET_SUBSET_2008-2014_5-7\AMF_US-SRC_FLUXNET_SUBSET_HH_2008-2014_5-7.csv"
# out_csv = "landsat_meteo_match_Src.csv"

untarred= True

# Select 18:30 US-R*
# Select 18:00 US-SRC

target_time = pd.to_datetime("18:30:00")
target_time = (target_time - pd.Timedelta(hours=7)).time()

"""
Variable mapping from AmeriFlux to output CSV:
----------------------------------------------
TA_F     -> Temp_24   (Air Temperature, °C, daily)
RH       -> RH_24     (Relative Humidity, %, daily)
WS_F     -> Wind_24   (Wind Speed, m/s, daily)
SW_IN_F  -> Rs_24     (Incoming Shortwave Radiation, W/m², daily)

TA_F     -> Temp_inst (Air Temperature, °C, half-hourly at 18:30)
RH_fixed -> RH_inst   (Relative Humidity, %, half-hourly at 18:30)
WS_F     -> Wind_inst (Wind Speed, m/s, half-hourly at 18:30)
SW_IN_F  -> Rs_inst   (Incoming Shortwave Radiation, W/m², half-hourly at 18:30)
"""

# -------------------------------
# Step 1. Extract Landsat dates
# -------------------------------
def extract_landsat_datetime_from_tar(tar_path):
    """Extract DATE_ACQUIRED and SCENE_CENTER_TIME from a Landsat tar file."""
    date, time = None, None
    with tarfile.open(tar_path, "r") as tar:
        mtl_file = [m for m in tar.getnames() if m.endswith("_MTL.txt")]
        if not mtl_file:
            return None, None
        mtl_file = mtl_file[0]
        f = tar.extractfile(mtl_file)
        if f is None:
            return None, None
        content = f.read().decode("utf-8")

        date_match = re.search(r"DATE_ACQUIRED\s=\s(.*)", content)
        time_match = re.search(r"SCENE_CENTER_TIME\s=\s(.*)", content)

        date = date_match.group(1).strip() if date_match else None
        time = time_match.group(1).strip().replace("Z", "") if time_match else None
    return date, time

def extract_landsat_datetime_from_folder(scene_dir):
    """Extract DATE_ACQUIRED and SCENE_CENTER_TIME from a Landsat folder (untarred)."""
    mtl_files = [f for f in os.listdir(scene_dir) if f.endswith("_MTL.txt")]
    if not mtl_files:
        return None, None
    mtl_path = os.path.join(scene_dir, mtl_files[0])

    with open(mtl_path, "r") as f:
        content = f.read()

    date_match = re.search(r"DATE_ACQUIRED\s=\s(.*)", content)
    time_match = re.search(r"SCENE_CENTER_TIME\s=\s(.*)", content)

    date = date_match.group(1).strip() if date_match else None
    time = time_match.group(1).strip().replace("Z", "") if time_match else None
    return date, time

# -------------------------------
# Build Landsat records
# -------------------------------
landsat_records = []
if untarred:
    # Loop over folders
    scene_dirs = [os.path.join(input_root, d) for d in os.listdir(input_root) if os.path.isdir(os.path.join(input_root, d))]
    for scene_dir in scene_dirs:
        date, time = extract_landsat_datetime_from_folder(scene_dir)
        if date:
            landsat_records.append({
                "scene": os.path.basename(scene_dir),
                "date": date,
                "time-utc": time
            })
else:
    # Loop over tar files
    tar_files = [os.path.join(input_root, f) for f in os.listdir(input_root) if f.endswith(".tar")]
    for tar_path in tar_files:
        date, time = extract_landsat_datetime_from_tar(tar_path)
        if date:
            landsat_records.append({
                "scene": os.path.basename(tar_path).replace(".tar", ""),  # strip ".tar"
                "date": date,
                "time-utc": time
            })

landsat_df = pd.DataFrame(landsat_records)
landsat_df["date"] = pd.to_datetime(landsat_df["date"])

# -------------------------------
# Step 2. Load AmeriFlux Daily dataset
# -------------------------------
flux_df = pd.read_csv(flux_daily_path)
flux_df["date"] = pd.to_datetime(flux_df["TIMESTAMP"], format="%Y%m%d")

# -------------------------------
# Step 3. Compute RH (daily)
# -------------------------------
def compute_rh(Ta, VPD_hPa):
    """Compute RH (%) from temperature (°C) and vapor pressure deficit (hPa)."""
    if pd.isna(Ta) or pd.isna(VPD_hPa):
        return np.nan
    VPD_kPa = VPD_hPa / 10.0  # convert hPa → kPa
    es = 0.6108 * np.exp((17.27 * Ta) / (Ta + 237.3))  # saturation vapor pressure (kPa)
    rh = 100 * (1 - VPD_kPa / es)
    return max(min(rh, 100), 0)

flux_df["RH"] = flux_df.apply(lambda row: compute_rh(row["TA_F"], row["VPD_F"]), axis=1)
flux_df["RH"] = flux_df["RH"].round(3)

subset_flux = flux_df[["date", "TA_F", "RH", "WS_F", "SW_IN_F"]]

# -------------------------------
# Step 4. Merge Landsat with Daily Flux
# -------------------------------
merged_df = pd.merge(landsat_df, subset_flux, on="date", how="left")
merged_df = merged_df.rename(columns={
    "TA_F": "Temp_24",
    "RH": "RH_24",
    "WS_F": "Wind_24",
    "SW_IN_F": "Rs_24"
})

# Round numeric values
for col in ["Temp_24", "RH_24", "Wind_24", "Rs_24"]:
    merged_df[col] = merged_df[col].round(3)

# -------------------------------
# Step 5. Load AmeriFlux Half-Hourly dataset for Instantaneous 18:30
# -------------------------------
hh_df = pd.read_csv(flux_halfhourly_path)
hh_df["datetime"] = pd.to_datetime(hh_df["TIMESTAMP_START"], format="%Y%m%d%H%M")
hh_df["date"] = hh_df["datetime"].dt.date
hh_df["time"] = hh_df["datetime"].dt.time

# Fix RH if missing
hh_df["RH_fixed"] = hh_df["RH"]
mask_invalid = (hh_df["RH"] < 0) | (hh_df["RH"].isna())
hh_df.loc[mask_invalid, "RH_fixed"] = hh_df.loc[mask_invalid].apply(
    lambda row: compute_rh(row["TA_F"], row["VPD_F"]), axis=1
)


hh_overpass = hh_df[hh_df["datetime"].dt.time == target_time].copy()
hh_overpass["date"] = pd.to_datetime(hh_overpass["datetime"].dt.date)

hh_overpass = hh_overpass[["date", "TA_F", "RH_fixed", "WS_F", "SW_IN_F"]]

# Rename
hh_overpass = hh_overpass.rename(columns={
    "TA_F": "Temp_inst",
    "RH_fixed": "RH_inst",
    "WS_F": "Wind_inst",
    "SW_IN_F": "Rs_inst"
})

for col in ["Temp_inst", "RH_inst", "Wind_inst", "Rs_inst"]:
    hh_overpass[col] = hh_overpass[col].round(3)

# -------------------------------
# Step 6. Merge Landsat with Instantaneous Flux
# -------------------------------
merged_df = pd.merge(merged_df, hh_overpass, on="date", how="left")

# -------------------------------
# Step 7. Add InputMap, OutputMap, and QualityMask columns
# -------------------------------
merged_df["InputMap"] = merged_df["scene"].apply(
    lambda s: os.path.join(input_root, s)
)
merged_df["OutputMap"] = merged_df["scene"].apply(
    lambda s: os.path.join(output_root, s)
)
merged_df["QualityMask"] = merged_df["scene"].apply(
    lambda s: os.path.join(input_root, s, f"{s}_QA_PIXEL.TIF")
)

# # -------------------------------
# # Step 8. Add placeholder Transm_24 (if needed)
# # -------------------------------
# if "Transm_24" not in merged_df.columns:
#     merged_df["Transm_24"] = np.nan

# -------------------------------
# Step 9. Reorder columns
# -------------------------------
col_order = [
    "scene", "date", "time-utc",          # keep identifiers first
    "Temp_inst", "Temp_24",
    "RH_inst", "RH_24",
    "Wind_inst", "Wind_24",
    "Rs_24", "Rs_inst",
    "InputMap", "OutputMap", "QualityMask"
]

merged_df = merged_df[[c for c in col_order if c in merged_df.columns]]

# -------------------------------
# Step 10. Save results
# -------------------------------
merged_df.to_csv(out_csv, index=False)
print(f"Saved results to {out_csv}")
print(merged_df.head())

