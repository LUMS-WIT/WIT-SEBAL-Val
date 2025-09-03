"""
2b.
This code extracts SEBAL (Surface Energy Balance Algorithm for Land) variables from SEBAL_out folder at user defined locations.

Returns
-------
pd.DataFrame
    A dataframe containing the extracted SEBAL variables for each site and date.
    sebal_extract_multisite.xlsx

"""

import os
import glob
import rasterio
import numpy as np
import pandas as pd
from rasterio.warp import transform

# ---------------------------------
# Define multiple sites here
# ---------------------------------
sites = [
    {
        "name": "RIs",
        "LAT": 43.1439,
        "LON": -116.736,
        "root_dir": r"F:\SEBAL\datasets\SEBAL_out\calibrations\Rls"
    },
    {
        "name": "Rms",
        "LAT": 43.0645,
        "LON": -116.7486,
        "root_dir": r"F:\SEBAL\datasets\SEBAL_out\calibrations\Rms"
    },

    {
        "name": "Rwf",
        "LAT": 43.1207,
        "LON": -116.7231,
        "root_dir": r"F:\SEBAL\datasets\SEBAL_out\calibrations\Rwf"
    },

    {
        "name": "Src",
        "LAT": 31.9083,
        "LON": -110.8395,
        "root_dir": r"F:\SEBAL\datasets\SEBAL_out\calibrations\Src"
    },
    # add more sites as needed
]

out_xlsx = "sebal_extract_multisite.xlsx"

# ---------------------------------
# Function to extract pixel + window
# ---------------------------------
def extract_pixel_and_window(tif_path, lat, lon, window_size=3):
    with rasterio.open(tif_path) as src:
        # Convert lat/lon -> row/col
        x, y = transform("EPSG:4326", src.crs, [lon], [lat])
        row, col = src.index(x[0], y[0])

        # Single pixel
        single_val = src.read(1)[row, col]

        # Window
        half = window_size // 2
        window = src.read(1)[row-half:row+half+1, col-half:col+half+1]
        if window.size > 0:
            mean_val = np.nanmean(np.where(window == src.nodata, np.nan, window))
        else:
            mean_val = np.nan

    if single_val == src.nodata:
        single_val = np.nan
    return single_val, mean_val

# ---------------------------------
# Loop over sites
# ---------------------------------
all_results = {}

for site in sites:
    LAT, LON = site["LAT"], site["LON"]
    root_dir = site["root_dir"]
    site_name = site["name"]

    records = []
    for scene_dir in glob.glob(os.path.join(root_dir, "*")):
        if not os.path.isdir(scene_dir):
            continue
        scene = os.path.basename(scene_dir)
        date = scene.split("_")[3]
        date = pd.to_datetime(date, format="%Y%m%d", errors="coerce")

        # File paths
        g_file  = glob.glob(os.path.join(scene_dir, "Output_energy_balance", "*G_inst*.tif"))
        le_file = glob.glob(os.path.join(scene_dir, "Output_energy_balance", "*LEinst*.tif"))
        rn_file = glob.glob(os.path.join(scene_dir, "Output_energy_balance", "*Rn_inst*.tif"))

        if not any([g_file, le_file, rn_file]):
            continue

        def safe_extract(path):
            if path:
                return extract_pixel_and_window(path[0], LAT, LON, window_size=5)
            return (np.nan, np.nan)

        G_pix, G_avg   = safe_extract(g_file)
        LE_pix, LE_avg = safe_extract(le_file)
        Rn_pix, Rn_avg = safe_extract(rn_file)

        # Derived terms
        H_pix = Rn_pix - G_pix - LE_pix if not np.isnan(Rn_pix) and not np.isnan(G_pix) and not np.isnan(LE_pix) else np.nan
        H_avg = Rn_avg - G_avg - LE_avg if not np.isnan(Rn_avg) and not np.isnan(G_avg) and not np.isnan(LE_avg) else np.nan

        Bowen_pix = H_pix / LE_pix if not np.isnan(H_pix) and not np.isnan(LE_pix) and LE_pix != 0 else np.nan
        Bowen_avg = H_avg / LE_avg if not np.isnan(H_avg) and not np.isnan(LE_avg) and LE_avg != 0 else np.nan

        precision = 3
        records.append({
            "scene": scene,
            "date": date,
            "G_inst_pix": None if np.isnan(G_pix) else round(G_pix, precision),
            "G_inst_avg": None if np.isnan(G_avg) else round(G_avg, precision),
            "LE_inst_pix": None if np.isnan(LE_pix) else round(LE_pix, precision),
            "LE_inst_avg": None if np.isnan(LE_avg) else round(LE_avg, precision),
            "Rn_inst_pix": None if np.isnan(Rn_pix) else round(Rn_pix, precision),
            "Rn_inst_avg": None if np.isnan(Rn_avg) else round(Rn_avg, precision),
            "H_inst_pix": None if np.isnan(H_pix) else round(H_pix, precision),
            "H_inst_avg": None if np.isnan(H_avg) else round(H_avg, precision),
            "Bowen_pix": None if np.isnan(Bowen_pix) else round(Bowen_pix, precision),
            "Bowen_avg": None if np.isnan(Bowen_avg) else round(Bowen_avg, precision),
        })

    df_site = pd.DataFrame(records)
    all_results[site_name] = df_site

# ---------------------------------
# Save to multi-sheet Excel
# ---------------------------------
with pd.ExcelWriter(out_xlsx, engine="xlsxwriter") as writer:
    for site_name, df_site in all_results.items():
        df_site.to_excel(writer, sheet_name=site_name, index=False)

print(f"✅ Multi-site extraction saved to {out_xlsx}")
