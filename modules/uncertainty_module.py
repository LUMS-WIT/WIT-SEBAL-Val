import os
import shutil
from pathlib import Path
import pandas as pd
import numpy as np

from utils import (
    SoilMoistureData,
    SebalSoilMoistureData,
    remove_nan_entries,
    save_to_excel,
    save_metadata,
    sebal_uncertainty_analysis,
    compute_statistics,
    plot_uncertainty_boxplot,
    compute_coverage_probability,
    uncertainty_signal_ratio,
    plot_uncertainty_distribution,
    plot_gpi_uncertainty,
    plot_relative_uncertainty_vs_sm,
    plot_coverage_probability,
)

from scaling import scaling, temporal_matching, temporal_matching_windowed
from sms_calibration import sms_calibrations


# -------------------------
# Local helpers (NO dependency on validation module)
# -------------------------
def raster_folder(raster_base: str, member: str, row_path: str) -> str:
    return fr"{raster_base}/{member}/{row_path}/"


def uq_member_dir(uq_base: Path, member: str, row_path: str, temporal_win: int) -> Path:
    return uq_base / member / f"{row_path}_{temporal_win}"


def uq_combine_dir(uq_base: Path, row_path: str, temporal_win: int) -> Path:
    return uq_base / "combine" / f"{row_path}_{temporal_win}"


def uq_rows_combined_dir(uq_base: Path, row_a: str, row_b: str, temporal_win: int) -> Path:
    # folder that holds combined-over-rows combined-UQ files
    return uq_base / "combine" / f"{row_a}_{row_b}_{temporal_win}"


def uq_metadata_file(uq_base: Path, member: str, row_path: str, temporal_win: int) -> Path:
    return uq_base / member / f"metadata_{row_path}_tw_{temporal_win}.xlsx"


def tw_to_window_params(tw: int):
    if tw == 0:
        return 0, 1
    if tw == 3:
        return 1, 2
    if tw == 5:
        return 2, 3
    if tw == 7:
        return 3, 4
    raise ValueError(f"Unsupported TEMPORAL_WIN={tw}. Use one of: 0, 3, 5, 7.")


# -------------------------
# Step 1: generate overlaps for a given member (mean/lower/upper)
# -------------------------
def generate_uq_overlaps_for_member(
    *,
    row_path: str,
    member: str,
    wit_sms_path: str,
    raster_folder_path: str,
    out_dir: Path,
    metadata_out: Path,
    temporal_win: int,
    rescaling: bool,
):
    out_dir.mkdir(parents=True, exist_ok=True)

    soil_moisture_data = SoilMoistureData(wit_sms_path)
    soil_moisture_data.read_data()
    metadata = soil_moisture_data.get_metadata()

    raster_data = SebalSoilMoistureData(raster_folder_path, pattern="Root_zone_moisture")

    validation_metadata = []
    print(f"[UQ overlaps] member={member} row={row_path} tw={temporal_win}")

    for _metadata in metadata:
        lat, lon = float(_metadata["latitude"]), float(_metadata["longitude"])

        csv_dates, csv_values = soil_moisture_data.get_soil_moisture_by_location(lat, lon)
        raster_dates, raster_values = raster_data.get_data(lat, lon)

        if raster_dates is None or raster_values is None:
            continue

        csv_dates, csv_values = remove_nan_entries(csv_dates, csv_values)
        raster_dates, raster_values = remove_nan_entries(raster_dates, raster_values)

        sensor_data = sms_calibrations((csv_dates, csv_values))
        model_data = (raster_dates, raster_values)

        if rescaling:
            model_data, _ = scaling(model_data, sensor_data)

        if temporal_win == 0:
            common_dates, ref_values, test_values, overlap_count = temporal_matching(
                model_data, sensor_data, temporal_win
            )
        else:
            half_window, min_valid = tw_to_window_params(temporal_win)
            (
                common_dates,
                ref_values,
                test_values,
                overlap_count,
                _n_used,
                _used_dates,
            ) = temporal_matching_windowed(
                model_data=model_data,
                sensor_data=sensor_data,
                half_window=half_window,
                min_valid=min_valid,
                agg_func=np.nanmean,
            )

        _metadata["overlaps"] = overlap_count
        if overlap_count > 0:
            validation_metadata.append(_metadata)

            data_to_save = list(zip(common_dates, ref_values, test_values))
            headers = ["Timestamp", "wit_sm", "sebal_sm"]

            # identical filenames across mean/lower/upper (required)
            fname = f"sebal_{row_path}_witgpi_{_metadata['gpi']}_lat_{lat}_lon_{lon}.xlsx"
            save_to_excel(data_to_save, str(out_dir / fname), headers)

    save_metadata(validation_metadata, str(metadata_out))
    print(f"[UQ overlaps] member={member} row={row_path} sensors={len(validation_metadata)}")


# -------------------------
# Step 2: combine mean/lower/upper into COMBINE_DIR
# -------------------------
def combine_uq(
    *,
    mean_dir: Path,
    lower_dir: Path,
    upper_dir: Path,
    combine_dir: Path,
):
    combine_dir.mkdir(parents=True, exist_ok=True)

    for mean_file in mean_dir.glob("*.xlsx"):
        fname = mean_file.name
        lower_file = lower_dir / fname
        upper_file = upper_dir / fname

        if not lower_file.exists() or not upper_file.exists():
            print(f"[UQ combine] Skipping {fname} (missing lower/upper file)")
            continue

        df_mean = pd.read_excel(mean_file)
        df_lower = pd.read_excel(lower_file)
        df_upper = pd.read_excel(upper_file)

        if not (
            df_mean["Timestamp"].equals(df_lower["Timestamp"])
            and df_mean["Timestamp"].equals(df_upper["Timestamp"])
        ):
            print(f"[UQ combine] Timestamp mismatch in {fname}, skipping")
            continue

        df_combined = df_mean.copy()
        df_combined["sebal_sm_l"] = df_lower["sebal_sm"]
        df_combined["sebal_sm_u"] = df_upper["sebal_sm"]

        if pd.api.types.is_datetime64_any_dtype(df_combined["Timestamp"]):
            df_combined["Timestamp"] = df_combined["Timestamp"].dt.date

        out_file = combine_dir / fname
        df_combined.to_excel(out_file, index=False)
        print(f"[UQ combine] Saved: {out_file}")


# -------------------------
# NEW: Step 2b: combine two ROWS' combine folders into one
# (no need to touch utils.sebal_uncertainty_analysis)
# -------------------------
def combine_uq_rows(
    *,
    combine_dir_a: Path,
    combine_dir_b: Path,
    out_dir: Path,
):
    """
    Copies all *.xlsx from combine_dir_a and combine_dir_b into out_dir.
    If a filename collision occurs, suffix it to avoid overwriting.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    def _copy_all(src_dir: Path):
        for f in src_dir.glob("*.xlsx"):
            dst = out_dir / f.name
            if dst.exists():
                stem = f.stem
                suffix = f.suffix
                dst = out_dir / f"{stem}__dup{suffix}"
            shutil.copy(f, dst)

    _copy_all(combine_dir_a)
    _copy_all(combine_dir_b)

    print(f"[UQ rows combine] Combined rows into: {out_dir}")


# -------------------------
# Step 3: run SEBAL uncertainty analysis on combine folder
# -------------------------
def run_uq_analysis(
    *,
    input_folder: Path,
    results_dir: Path,
    output_basename: str,
):
    """
    output_basename examples:
      - "SEBAL_uncertainty_summary_149039_tw_0"
      - "SEBAL_uncertainty_summary_tw_0"   (combined rows)
    """
    results_dir.mkdir(parents=True, exist_ok=True)

    uq_dict, uq_values = sebal_uncertainty_analysis(str(input_folder))
    uq_stats = compute_statistics(uq_dict)

    print("------------- SEBAL model uncertainty ----------------")
    print(uq_stats)

    plot_uncertainty_boxplot(uq_dict)

    coverage = compute_coverage_probability(uq_values)
    print("\nCoverage Probability at 95% confidence interval:", coverage)

    usr = uncertainty_signal_ratio(uq_values)
    print("Uncertainty Signal Ratio:", usr)

    uq_stats_df = pd.DataFrame(uq_stats).T
    uq_stats_df.index.name = "Metric"
    uq_stats_df["USR"] = usr
    uq_stats_df["Coverage"] = coverage

    out_excel = results_dir / f"{output_basename}.xlsx"
    uq_stats_df.to_excel(out_excel)
    print(f"[UQ] Summary saved -> {out_excel}")

    plot_uncertainty_distribution(uq_values)
    plot_gpi_uncertainty(uq_values)
    plot_relative_uncertainty_vs_sm(uq_values)
    plot_coverage_probability(uq_values)