"""
# 3. Merges SEBAL extraction and flux tower data for calibration analysis

returns:
merged_calibration_multisite.csv
"""

import pandas as pd

# -----------------------------
# Inputs
# -----------------------------
sebal_xlsx = r"sebal_extract_multisite.xlsx"
flux_xlsx  = r"landsat_meteo_energy_multisite.xlsx"
out_csv    = r"merged_calibration_multisite.csv"

# -----------------------------
# Load available sheets
# -----------------------------
sebal_sheets = pd.ExcelFile(sebal_xlsx).sheet_names
flux_sheets  = pd.ExcelFile(flux_xlsx).sheet_names

print("SEBAL sheets:", sebal_sheets)
print("Flux sheets:", flux_sheets)

# -----------------------------
# Collect merged per site
# -----------------------------
all_calibs = []

for sheet in sebal_sheets:
    site = sheet  # same site name convention
    
    # SEBAL data
    df_sebal = pd.read_excel(sebal_xlsx, sheet_name=sheet)
    df_sebal["date"] = pd.to_datetime(df_sebal["date"])
    
    # Flux data (EnergyFluxes sheet for this site)
    flux_sheet = f"{site}_EnergyFluxes"
    if flux_sheet not in flux_sheets:
        print(f"⚠️ Skipping {site}, no flux sheet found")
        continue
    df_flux = pd.read_excel(flux_xlsx, sheet_name=flux_sheet)
    df_flux["date"] = pd.to_datetime(df_flux["date"])
    
    # Merge
    df_merged = pd.merge(
        df_sebal,
        df_flux,
        on=["scene","date"],
        how="inner"
    )
    
    # Calibration columns
    calib_cols = [
        "scene", "date",
        "Rn_inst_pix", "Rn_inst_avg",
        "G_inst_pix", "G_inst_avg",
        "LE_inst_pix", "LE_inst_avg",
        "EF_inst_pix", "EF_inst_avg",
        "H_inst_pix", "H_inst_avg",
        "Bowen_pix", "Bowen_avg",
        "NETRAD", "G_F_MDS", "H_F_MDS", "LE_F_MDS",
        "Bowen_ratio", "EF"
    ]
    df_calib = df_merged[[c for c in calib_cols if c in df_merged.columns]].copy()
    
    # Round
    num_cols = df_calib.select_dtypes(include="number").columns
    df_calib[num_cols] = df_calib[num_cols].round(3)
    
    # Add site column
    df_calib["site"] = site
    all_calibs.append(df_calib)

# -----------------------------
# Save merged calibration file
# -----------------------------
df_all = pd.concat(all_calibs, ignore_index=True)
df_all.to_csv(out_csv, index=False)

print(f"✅ Merged calibration dataset saved to {out_csv}")
print(df_all.head())
