import os
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

from utils import (
    SoilMoistureData,
    SebalSoilMoistureData,
    remove_nan_entries,
    save_to_excel,
    save_metadata,
    save_to_plot,
    validations_gpi_adv,
    compute_statistics,
    plot_box_and_whiskers,
    plot_metric_with_ci,
)

from scaling import scaling, temporal_matching
from sms_calibration import sms_calibrations


# -------------------------
# Path helpers
# -------------------------
def raster_folder(raster_base: str, member: str, row_path: str) -> str:
    return fr"{raster_base}/{member}/{row_path}/"


def validation_folder(val_base: Path, member: str, row_path: str, temporal_win: int) -> Path:
    return val_base / member / f"{row_path}_{temporal_win}"


def metadata_file(val_base: Path, member: str, row_path: str, temporal_win: int) -> Path:
    return val_base / member / f"metadata_{row_path}_tw_{temporal_win}.xlsx"


def images_folder(fig_base: Path, member: str, row_path: str) -> Path:
    # per-site time-series plots (SAVE_PLOT=True)
    return fig_base / member / row_path


def figs_row_folder(fig_base: Path, row_path: str) -> Path:
    # validation plots for a row: figs/149039, figs/150039
    return fig_base / row_path


def results_file(results_base: Path, name: str, temporal_win: int) -> Path:
    return results_base / f"{name}_tw_{temporal_win}.xlsx"


# -------------------------
# Listing + metadata
# -------------------------
def list_validation_files(folder: Path) -> list[str]:
    if not folder.exists():
        return []
    return [
        str(folder / f)
        for f in os.listdir(folder)
        if f.endswith(".xlsx") and "witgpi" in f
    ]


def list_validation_files_multi(folders: list[Path]) -> list[str]:
    files: list[str] = []
    for folder in folders:
        files.extend(list_validation_files(folder))
    return files


def read_metadata_df(metadata_xlsx: Path) -> pd.DataFrame:
    df = pd.read_excel(metadata_xlsx)
    df["gpi"] = df["gpi"].astype(str)
    return df


def merge_metadata(df_a: pd.DataFrame, df_b: pd.DataFrame) -> pd.DataFrame:
    out = pd.concat([df_a, df_b], ignore_index=True)
    out["gpi"] = out["gpi"].astype(str)
    return out


# -------------------------
# Plotting (paired: always DISPLAY + SAVE)
# -------------------------
def make_paired_scatter_figure(paired_values: pd.DataFrame, stats_results: dict, title: str):
    x = paired_values["sebal_sm"].to_numpy(dtype=float)
    y = paired_values["wit_sm"].to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    ax.scatter(x, y, s=12, alpha=0.5, edgecolors="none")

    if x.size and y.size:
        min_v = float(min(x.min(), y.min()))
        max_v = float(max(x.max(), y.max()))
        ax.plot([min_v, max_v], [min_v, max_v], "--", color="gray", linewidth=1)

    ax.set_xlabel("SEBAL-derived Soil Moisture (m³/m³)")
    ax.set_ylabel("WITSMS Soil Moisture (m³/m³)")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal", adjustable="box")

    mean_bias = stats_results["bias"]["mean"]
    mean_mse = stats_results["mse"]["mean"]
    mean_p = stats_results["p_rho"]["mean"]
    mean_s = stats_results["s_rho"]["mean"]
    mean_ubr = stats_results["ubrmsd"]["mean"]

    rmse_disp = (mean_mse ** 0.5) if mean_mse is not None else float("nan")
    textstr = (
        f"Bias = {mean_bias:.3f} m³/m³\n"
        f"RMSE = {rmse_disp:.3f} m³/m³\n"
        f"ubRMSD = {mean_ubr:.3f} m³/m³\n"
        f"R = {mean_p:.3f}\n"
        fr"$\rho$ = {mean_s:.3f}"
        f"\nN = {len(paired_values)}"
    )
    ax.text(
        0.02, 0.98, textstr,
        transform=ax.transAxes,
        va="top", ha="left",
        bbox=dict(facecolor="white", edgecolor="gray", alpha=0.9),
    )

    fig.tight_layout()
    return fig


def save_fig(fig, out_png: Path):
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    print(f"[plot] saved -> {out_png}")


# -------------------------
# Step 1: overlaps
# -------------------------
def generate_overlaps(
    *,
    row_path: str,
    member: str,
    wit_sms_path: str,
    raster_folder_path: str,
    validation_folder_path: Path,
    images_folder_path: Path,
    metadata_file_path: Path,
    temporal_win: int,
    rescaling: bool,
    save_plot: bool,
):
    validation_folder_path.mkdir(parents=True, exist_ok=True)
    images_folder_path.mkdir(parents=True, exist_ok=True)

    soil_moisture_data = SoilMoistureData(wit_sms_path)
    soil_moisture_data.read_data()
    metadata = soil_moisture_data.get_metadata()

    raster_data = SebalSoilMoistureData(raster_folder_path, pattern="Root_zone_moisture")

    validation_metadata = []
    print(f"[overlaps] member={member} row={row_path} tw={temporal_win}")

    for _metadata in metadata:
        lat, lon = float(_metadata["latitude"]), float(_metadata["longitude"])

        csv_dates, csv_values = soil_moisture_data.get_soil_moisture_by_location(lat, lon)
        raster_dates, raster_values = raster_data.get_data(lat, lon)

        if raster_dates is None or raster_values is None:
            continue

        csv_dates, csv_values = remove_nan_entries(csv_dates, csv_values)
        raster_dates, raster_values = remove_nan_entries(raster_dates, raster_values)

        ref_data = sms_calibrations((csv_dates, csv_values))
        test_data = (raster_dates, raster_values)

        if rescaling:
            test_data, _ = scaling(test_data, ref_data)
            csv_dates, csv_values = ref_data
            raster_dates, raster_values = test_data

        common_dates, ref_values, test_values, overlap_count = temporal_matching(
            test_data, ref_data, temporal_win
        )

        _metadata["overlaps"] = overlap_count
        if overlap_count > 0:
            validation_metadata.append(_metadata)

            data_to_save = list(zip(common_dates, ref_values, test_values))
            headers = ["Timestamp", "wit_sm", "sebal_sm"]
            fname = f"sebal_{row_path}_{member}_witgpi_{_metadata['gpi']}_lat_{lat}_lon_{lon}.xlsx"
            save_to_excel(data_to_save, str(validation_folder_path / fname), headers)

            if save_plot:
                img_name = f"sebal_{row_path}_{member}_witgpi_{_metadata['gpi']}_lat_{lat}_lon_{lon}.png"
                save_to_plot(
                    csv_dates, csv_values,
                    raster_dates, raster_values,
                    lat, lon,
                    str(images_folder_path / img_name),
                )

    save_metadata(validation_metadata, str(metadata_file_path))
    print(f"[overlaps] member={member} row={row_path} sensors={len(validation_metadata)}")


# -------------------------
# Step 2: validations (paired always DISPLAY + SAVE)
# -------------------------
def run_validations_and_save_all(
    *,
    tag: str,
    files: list[str],
    metadata_df: pd.DataFrame,
    output_xlsx: Path,
    figs_out_dir: Path,
    outlier_threshold: float,
    show_all_plots: bool,
    save_all_plots: bool,
    temporal_win: int,
):
    metrics_dict, num_of_obs, paired_values = validations_gpi_adv(
        files=files,
        threshold=outlier_threshold,
    )
    stats_results = compute_statistics(metrics_dict)

    # Write workbook
    metrics_df = pd.DataFrame(metrics_dict)
    metrics_df["gpi"] = metrics_df["gpi"].astype(str)
    merged_df = pd.merge(metadata_df, metrics_df, on="gpi", how="left")

    summary_data = {
        "Metric": ["bias", "mse", "ubrmsd", "p_rho", "s_rho"],
        "mean": [
            stats_results["bias"]["mean"],
            stats_results["mse"]["mean"],
            stats_results["ubrmsd"]["mean"],
            stats_results["p_rho"]["mean"],
            stats_results["s_rho"]["mean"],
        ],
        "median": [
            stats_results["bias"]["median"],
            stats_results["mse"]["median"],
            stats_results["ubrmsd"]["median"],
            stats_results["p_rho"]["median"],
            stats_results["s_rho"]["median"],
        ],
        "IQR": [
            stats_results["bias"]["IQR"],
            stats_results["mse"]["IQR"],
            stats_results["ubrmsd"]["IQR"],
            stats_results["p_rho"]["IQR"],
            stats_results["s_rho"]["IQR"],
        ],
    }
    summary_df = pd.DataFrame(summary_data)
    obs_df = pd.DataFrame({"Metric": ["Observations"], "mean": [num_of_obs], "median": [""], "IQR": [""]})
    summary_df = pd.concat([obs_df, summary_df], ignore_index=True)

    output_xlsx.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(output_xlsx, engine="xlsxwriter") as writer:
        merged_df.to_excel(writer, sheet_name="MetaData", index=False)
        summary_df.to_excel(writer, sheet_name="Summary", index=False)

    print(f"[validate] {tag} saved -> {output_xlsx}")

    figs_out_dir.mkdir(parents=True, exist_ok=True)

    # Paired scatter: ALWAYS DISPLAY + SAVE
    paired_png = figs_out_dir / f"paired_scatter_tw_{temporal_win}.png"
    fig = make_paired_scatter_figure(paired_values, stats_results, title=tag)
    save_fig(fig, paired_png)
    plt.show()          # display
    plt.close(fig)      # close after display to avoid memory buildup

    # Extra plots:
    # - show_all_plots -> display
    # - save_all_plots -> save to disk (even if not displaying)
    if save_all_plots:
        plot_box_and_whiskers(metrics_dict, filename=str(figs_out_dir / f"boxplot_{tag}.png"), save=True, show=show_all_plots)
        plot_metric_with_ci(metrics_dict, metric="ubrmsd",
                            filename=str(figs_out_dir / f"ci_ubrmsd_{tag}.png"), save=True, show=show_all_plots)
        plot_metric_with_ci(metrics_dict, metric="bias",
                            filename=str(figs_out_dir / f"ci_bias_{tag}.png"), save=True, show=show_all_plots)
    elif show_all_plots:
        # show only (no saving)
        plot_box_and_whiskers(metrics_dict, save=False, show=True)
        plot_metric_with_ci(metrics_dict, metric="ubrmsd", save=False, show=True)
        plot_metric_with_ci(metrics_dict, metric="bias", save=False, show=True)

    return metrics_dict, num_of_obs, paired_values, stats_results