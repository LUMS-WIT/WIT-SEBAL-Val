"""Match daily and half-hourly tower records to available model scenes."""
import os
import glob
import re
import tarfile
from pathlib import Path
import pandas as pd
import numpy as np

def compute_rh(Ta, VPD_hPa):
    if pd.isna(Ta) or pd.isna(VPD_hPa): return np.nan
    VPD_kPa = VPD_hPa / 10.0
    es = 0.6108 * np.exp((17.27 * Ta) / (Ta + 237.3))
    rh = 100 * (1 - VPD_kPa / es)
    return max(min(rh, 100), 0)

def extract_landsat_datetime_from_tar(tar_path):
    date, time = None, None
    with tarfile.open(tar_path, "r") as tar:
        mtl_file = [m for m in tar.getnames() if m.endswith("_MTL.txt")]
        if not mtl_file: return None, None
        f = tar.extractfile(mtl_file[0])
        if f is None: return None, None
        content = f.read().decode("utf-8")
        dm = re.search(r"DATE_ACQUIRED\s=\s(.*)", content)
        tm = re.search(r"SCENE_CENTER_TIME\s=\s(.*)", content)
        date = dm.group(1).strip() if dm else None
        time = tm.group(1).strip().replace("Z","") if tm else None
    return date, time

# ---------------------------------------
# Process each site
# ---------------------------------------

def run(sites, out_excel):
    Path(out_excel).parent.mkdir(parents=True, exist_ok=True)
    sites = [dict(s, target_time=pd.Timestamp(s['target_time']).time())
             if isinstance(s['target_time'], str) else s for s in sites]
    with pd.ExcelWriter(out_excel, engine="xlsxwriter") as writer:
        for site in sites:
            name           = site["name"]
            flux_daily     = site["flux_daily_path"]
            flux_halfhour  = site["flux_halfhourly_path"]
            input_root     = site.get("input_root", "")
            output_root    = site["output_root"]
            target_time    = site["target_time"]

            print(f"Processing {name} using scene folders in: {output_root}")

            # ============================================================
            # A) Build Landsat scene table FROM OUTPUT FOLDERS (preferred)
            # ============================================================
            landsat_records = []
            for scene_dir in glob.glob(os.path.join(output_root, "*")):
                if not os.path.isdir(scene_dir):
                    continue
                scene = os.path.basename(scene_dir)
                # Expect standard Landsat naming: LC08_L1TP_XXX_YYYYMMDD_...
                # date token is index 3 (0-based)
                toks = scene.split("_")
                if len(toks) >= 4:
                    dt = pd.to_datetime(toks[3], format="%Y%m%d", errors="coerce")
                else:
                    dt = pd.NaT
                landsat_records.append({
                    "scene": scene,
                    "date": dt,
                    "time": target_time.strftime("%H:%M:%S")  # stamp site-specific overpass time
                })

            landsat_df = pd.DataFrame(landsat_records).dropna(subset=["date"]).copy()

            # ============================================================
            # (Optional) Previous TAR-based method — kept for fallback
            # ============================================================
            # tar_files = [os.path.join(input_root, f) for f in os.listdir(input_root) if f.endswith(".tar")]
            # landsat_records = []
            # for tar_path in tar_files:
            #     date_str, time_str = extract_landsat_datetime_from_tar(tar_path)
            #     if date_str:
            #         landsat_records.append({
            #             "scene": os.path.basename(tar_path).replace(".tar",""),
            #             "date": pd.to_datetime(date_str),
            #             "time": time_str
            #         })
            # landsat_df = pd.DataFrame(landsat_records)

            # ---------------------------------------
            # Daily flux
            # ---------------------------------------
            flux_df = pd.read_csv(flux_daily)
            flux_df["date"] = pd.to_datetime(flx := flux_df["TIMESTAMP"], format="%Y%m%d")
            flux_df["RH"] = flux_df.apply(lambda r: compute_rh(r["TA_F"], r["VPD_F"]), axis=1).round(3)

            subset_flux = flux_df[["date","TA_F","RH","WS_F","SW_IN_F"]].copy()
            merged_df = pd.merge(landsat_df, subset_flux, on="date", how="left")
            merged_df = merged_df.rename(columns={
                "TA_F":"Temp_24","RH":"RH_24","WS_F":"Wind_24","SW_IN_F":"Rs_24"
            })
            for c in ["Temp_24","RH_24","Wind_24","Rs_24"]:
                merged_df[c] = merged_df[c].round(3)

            # ---------------------------------------
            # Half-hourly flux @ site-specific target_time
            # ---------------------------------------
            hh_df = pd.read_csv(flux_halfhour)
            hh_df["datetime"] = pd.to_datetime(hh_df["TIMESTAMP_START"], format="%Y%m%d%H%M")
            hh_df["RH_fixed"] = hh_df["RH"]
            mask_invalid = (hh_df["RH"] < 0) | (hh_df["RH"].isna())
            hh_df.loc[mask_invalid,"RH_fixed"] = hh_df.loc[mask_invalid].apply(
                lambda r: compute_rh(r["TA_F"], r["VPD_F"]), axis=1
            )

            hh_overpass = hh_df[hh_df["datetime"].dt.time == target_time].copy()
            hh_overpass["date"] = pd.to_datetime(hh_overpass["datetime"].dt.date)
            hh_overpass = hh_overpass.rename(columns={
                "TA_F":"Temp_inst","RH_fixed":"RH_inst","WS_F":"Wind_inst","SW_IN_F":"Rs_inst"
            })
            hh_overpass = hh_overpass[["date","Temp_inst","RH_inst","Wind_inst","Rs_inst"]].round(3)

            merged_df = pd.merge(merged_df, hh_overpass, on="date", how="left")

            # ---------------------------------------
            # Paths (keep if useful)
            # ---------------------------------------
            merged_df["InputMap"]  = merged_df["scene"].apply(lambda s: os.path.join(input_root, s))
            merged_df["OutputMap"] = merged_df["scene"].apply(lambda s: os.path.join(output_root, s))
            merged_df["QualityMask"] = merged_df["scene"].apply(
                lambda s: os.path.join(input_root, s, f"{s}_QA_PIXEL.TIF")
            )

            # ---------------------------------------
            # Energy balance terms @ target_time
            # ---------------------------------------
            hh_energy = hh_df[hh_df["datetime"].dt.time == target_time].copy()
            hh_energy["date"] = pd.to_datetime(hh_energy["datetime"].dt.date)
            energy_cols = ["date","NETRAD","G_F_MDS","H_F_MDS","LE_F_MDS"]
            energy_overpass = hh_energy[energy_cols].replace(-9999, np.nan).copy()
            energy_overpass["Bowen_ratio"] = energy_overpass["H_F_MDS"] / energy_overpass["LE_F_MDS"]
            energy_overpass["EF"] = energy_overpass["LE_F_MDS"] / (energy_overpass["NETRAD"] - energy_overpass["G_F_MDS"])
            energy_overpass["EF"] = energy_overpass["EF"].clip(0.0, 1.8)

            energy_df = pd.merge(landsat_df[["scene","date","time"]], energy_overpass, on="date", how="left")

            # ---------------------------------------
            # Save per-site sheets
            # ---------------------------------------
            merged_df.to_excel(writer, sheet_name=f"{name}_Meteo", index=False)
            energy_df.to_excel(writer, sheet_name=f"{name}_EnergyFluxes", index=False)

    print(f"✅ Multi-site tower extraction saved to {out_excel}")


    return out_excel
