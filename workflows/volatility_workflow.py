from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import config as cfg
from modules.volatility_module import (
    run_case,
    run_case_from_files,
    load_validation_metadata_metrics,
    maybe_add_delta_r,
)


def _case_out_dir(base: Path, raster_stat: str, row_path_label: str) -> Path:
    d = base / "inter_overpass" / raster_stat / row_path_label
    d.mkdir(parents=True, exist_ok=True)
    return d


def _save_xlsx(path: Path, sheet: str, df: pd.DataFrame):
    path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(path, engine="xlsxwriter") as w:
        df.to_excel(w, sheet_name=sheet, index=False)


def _scatter(df: pd.DataFrame, *, x: str, y: str, out_png: Path, xlabel: str, ylabel: str, title: str):
    d = df[[x, y]].replace([np.inf, -np.inf], np.nan).dropna()
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    ax.scatter(d[x], d[y], s=22, alpha=0.80, edgecolors="none")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.25)

    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=350, bbox_inches="tight")
    plt.show()
    plt.close(fig)


def _run_one_case_and_save(
    *,
    window_df: pd.DataFrame,
    site_df: pd.DataFrame,
    out_dir: Path,
    raster_stat: str,
    row_path_label: str,
    validations_results_dir: Path,
    validation_xlsx: Path | None,
    allow_delta_r: bool,
):
    _save_xlsx(out_dir / f"window_level_metrics_{raster_stat}_{row_path_label}.xlsx", "WindowMetrics", window_df)
    _save_xlsx(out_dir / f"site_level_metrics_{raster_stat}_{row_path_label}.xlsx", "SiteMetrics", site_df)

    merged_df = None
    if validation_xlsx is not None and validation_xlsx.exists() and not site_df.empty:
        try:
            vdf = load_validation_metadata_metrics(validation_xlsx)
            merged_df = site_df.copy()
            merged_df["gpi"] = merged_df["gpi"].astype(str)
            merged_df = merged_df.merge(vdf, on="gpi", how="left", suffixes=("", "_val"))

            if allow_delta_r:
                merged_df = maybe_add_delta_r(
                    merged_df,
                    validations_results_dir=validations_results_dir,
                    row_path=row_path_label,
                    baseline_tw=0,
                    candidate_tws=(3, 5, 7),
                )

            _save_xlsx(
                out_dir / f"site_level_with_validation_{raster_stat}_{row_path_label}.xlsx",
                "MergedSiteMetrics",
                merged_df,
            )
        except Exception as e:
            print(f"[volatility] merge validation metrics failed for {row_path_label}: {e}")

    # plots (need p_rho)
    if merged_df is not None and "p_rho" in merged_df.columns:
        _scatter(
            merged_df,
            x="median_AV_norm",
            y="p_rho",
            out_png=out_dir / f"scatter_AVnorm_vs_prho_{raster_stat}_{row_path_label}.png",
            xlabel=r"Median(AV$_{norm}$) [m$^3$/m$^3$ per day]",
            ylabel="Pearson r (tw=0)",
            title=f"{row_path_label} {raster_stat}: AV_norm vs Pearson r",
        )
        _scatter(
            merged_df,
            x="median_Missed_norm",
            y="p_rho",
            out_png=out_dir / f"scatter_Missednorm_vs_prho_{raster_stat}_{row_path_label}.png",
            xlabel=r"Median(Missed$_{norm}$) [-]",
            ylabel="Pearson r (tw=0)",
            title=f"{row_path_label} {raster_stat}: Missed_norm vs Pearson r",
        )
        _scatter(
            merged_df,
            x="median_QV_norm",
            y="p_rho",
            out_png=out_dir / f"scatter_QVnorm_vs_prho_{raster_stat}_{row_path_label}.png",
            xlabel=r"Median(QV$_{norm}$) [(m$^3$/m$^3$)$^2$ per day]",
            ylabel="Pearson r (tw=0)",
            title=f"{row_path_label} {raster_stat}: QV_norm vs Pearson r",
        )

    # terminal summary per case
    sites_processed = int(len(site_df)) if site_df is not None else 0
    valid_sites = int((site_df["number_of_windows"] > 0).sum()) if sites_processed else 0
    total_windows = int(len(window_df)) if window_df is not None else 0
    med_med_av = float(site_df["median_AV_norm"].median()) if sites_processed else float("nan")
    med_med_missed = float(site_df["median_Missed_norm"].median()) if sites_processed else float("nan")

    print("\n[volatility summary]")
    print(f"case: row={row_path_label}, raster_stat={raster_stat}")
    print(f"sites_processed={sites_processed}")
    print(f"valid_sites_with_windows={valid_sites}")
    print(f"total_windows={total_windows}")
    print(f"median(site median_AV_norm)={med_med_av:.6f}")
    print(f"median(site median_Missed_norm)={med_med_missed:.6f}")

    if merged_df is not None and "p_rho" in merged_df.columns:
        d = merged_df[["median_Missed_norm", "p_rho"]].replace([np.inf, -np.inf], np.nan).dropna()
        if len(d) >= 3:
            spearman = d["median_Missed_norm"].corr(d["p_rho"], method="spearman")
            print(f"Spearman(median_Missed_norm, p_rho)={spearman:.4f}")
        else:
            print("Spearman(median_Missed_norm, p_rho)=NaN (insufficient data)")


def main():
    if not getattr(cfg, "RUN_INTER_OVERPASS_DIAGNOSTICS", False):
        print("[volatility] RUN_INTER_OVERPASS_DIAGNOSTICS=False; skipping.")
        return

    overlap_base = cfg.VALIDATIONS_OUTPUT_BASE / "validation_points"
    validations_results_dir = cfg.VALIDATIONS_OUTPUT_BASE / "results"
    out_base = cfg.VOLATILITY_OUTPUT_BASE

    raster_stats = getattr(cfg, "INTER_OVERPASS_RASTER_STATS", ["mean"])
    min_valid = getattr(cfg, "MIN_VALID_INCREMENTS", 2)
    eps = getattr(cfg, "EPSILON_MISSED_VARIATION", 1e-9)

    baseline_tw = 0

    # -------------------------
    # 1) Per-row cases
    # -------------------------
    for row_path in cfg.ROW_PATHS:
        for raster_stat in raster_stats:
            overlap_folder = overlap_base / raster_stat / f"{row_path}_{baseline_tw}"
            if not overlap_folder.exists():
                print(f"[volatility] overlap folder not found: {overlap_folder}")
                continue

            window_df, site_df = run_case(
                overlap_folder=overlap_folder,
                wit_sms_path=cfg.WIT_SMS_PATH,
                row_path=row_path,
                raster_stat=raster_stat,
                min_valid_increments=min_valid,
                epsilon=eps,
            )

            out_dir = _case_out_dir(out_base, raster_stat, row_path)
            validation_xlsx = validations_results_dir / f"validations_{row_path}_tw_{baseline_tw}.xlsx"

            _run_one_case_and_save(
                window_df=window_df,
                site_df=site_df,
                out_dir=out_dir,
                raster_stat=raster_stat,
                row_path_label=row_path,
                validations_results_dir=validations_results_dir,
                validation_xlsx=validation_xlsx,
                allow_delta_r=True,
            )

    # -------------------------
    # 2) Combined (both rows together)
    # -------------------------
    if len(cfg.ROW_PATHS) >= 2:
        row_a = cfg.ROW_PATHS[0]
        row_b = cfg.ROW_PATHS[1]
        combined_label = f"combined_{row_a}_{row_b}"

        for raster_stat in raster_stats:
            folder_a = overlap_base / raster_stat / f"{row_a}_{baseline_tw}"
            folder_b = overlap_base / raster_stat / f"{row_b}_{baseline_tw}"
            if not folder_a.exists() or not folder_b.exists():
                print(f"[volatility] combined skip: missing {folder_a} or {folder_b}")
                continue

            overlap_files = sorted(list(folder_a.glob("*.xlsx")) + list(folder_b.glob("*.xlsx")))
            window_df, site_df = run_case_from_files(
                overlap_files=overlap_files,
                wit_sms_path=cfg.WIT_SMS_PATH,
                row_path_label=combined_label,
                raster_stat=raster_stat,
                min_valid_increments=min_valid,
                epsilon=eps,
            )

            out_dir = _case_out_dir(out_base, raster_stat, combined_label)

            # combined validation metrics file
            validation_xlsx = validations_results_dir / f"validations_tw_{baseline_tw}.xlsx"

            _run_one_case_and_save(
                window_df=window_df,
                site_df=site_df,
                out_dir=out_dir,
                raster_stat=raster_stat,
                row_path_label=combined_label,
                validations_results_dir=validations_results_dir,
                validation_xlsx=validation_xlsx,
                allow_delta_r=False,  # delta_r is row-specific; keep off for combined
            )


if __name__ == "__main__":
    main()