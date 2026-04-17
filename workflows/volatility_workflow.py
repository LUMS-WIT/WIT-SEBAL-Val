from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import config as cfg
from modules.volatility_module import run_case


def _case_out_dir(base: Path, raster_stat: str, row_path_label: str) -> Path:
    d = base / "inter_overpass" / raster_stat / row_path_label
    d.mkdir(parents=True, exist_ok=True)
    return d


def _save_xlsx(path: Path, sheet: str, df: pd.DataFrame):
    path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(path, engine="xlsxwriter") as writer:
        df.to_excel(writer, sheet_name=sheet, index=False)


def _scatter(
    df: pd.DataFrame,
    *,
    x: str,
    y: str,
    out_png: Path,
    xlabel: str,
    ylabel: str,
    title: str,
):
    d = df[[x, y]].replace([np.inf, -np.inf], np.nan).dropna()
    if d.empty:
        return

    fig, ax = plt.subplots(figsize=(6.5, 5.4))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    ax.scatter(d[x], d[y], s=26, alpha=0.85, edgecolors="none")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=350, bbox_inches="tight")
    plt.close(fig)


def _save_case_outputs(
    *,
    out_dir: Path,
    raster_stat: str,
    row_path_label: str,
    window_df: pd.DataFrame,
    site_df: pd.DataFrame,
):
    _save_xlsx(
        out_dir / f"window_level_metrics_{raster_stat}_{row_path_label}.xlsx",
        "WindowMetrics",
        window_df,
    )
    _save_xlsx(
        out_dir / f"site_level_metrics_{raster_stat}_{row_path_label}.xlsx",
        "SiteMetrics",
        site_df,
    )

    if cfg.VOLATILITY_SAVE_SCATTERS and not site_df.empty:
        if "p_rho_tw0" in site_df.columns:
            _scatter(
                site_df,
                x="median_AV_norm",
                y="p_rho_tw0",
                out_png=out_dir / f"scatter_AVnorm_vs_prho_{raster_stat}_{row_path_label}.png",
                xlabel=r"Median(AV$_{norm}$) [m$^3$/m$^3$ per day]",
                ylabel="Pearson r (tw=0)",
                title=f"{row_path_label} {raster_stat}: AV_norm vs Pearson r",
            )
            _scatter(
                site_df,
                x="median_Missed_norm",
                y="p_rho_tw0",
                out_png=out_dir / f"scatter_Missednorm_vs_prho_{raster_stat}_{row_path_label}.png",
                xlabel=r"Median(Missed$_{norm}$) [-]",
                ylabel="Pearson r (tw=0)",
                title=f"{row_path_label} {raster_stat}: Missed_norm vs Pearson r",
            )
            _scatter(
                site_df,
                x="median_QV_norm",
                y="p_rho_tw0",
                out_png=out_dir / f"scatter_QVnorm_vs_prho_{raster_stat}_{row_path_label}.png",
                xlabel=r"Median(QV$_{norm}$) [(m$^3$/m$^3$)$^2$ per day]",
                ylabel="Pearson r (tw=0)",
                title=f"{row_path_label} {raster_stat}: QV_norm vs Pearson r",
            )


def _print_summary(label: str, raster_stat: str, window_df: pd.DataFrame, site_df: pd.DataFrame):
    sites_processed = int(len(site_df)) if site_df is not None else 0
    valid_sites = int((site_df["number_of_windows"] > 0).sum()) if sites_processed else 0
    total_windows = int(len(window_df)) if window_df is not None else 0

    med_av = float(site_df["median_AV_norm"].median()) if sites_processed else np.nan
    med_missed = float(site_df["median_Missed_norm"].median(skipna=True)) if sites_processed else np.nan

    print("\n[volatility summary]")
    print(f"case: row={label}, raster_stat={raster_stat}")
    print(f"sites_processed={sites_processed}")
    print(f"valid_sites_with_windows={valid_sites}")
    print(f"total_windows={total_windows}")
    print(f"median(site median_AV_norm)={med_av:.6f}" if np.isfinite(med_av) else "median(site median_AV_norm)=NaN")
    print(f"median(site median_Missed_norm)={med_missed:.6f}" if np.isfinite(med_missed) else "median(site median_Missed_norm)=NaN")

    if site_df is not None and not site_df.empty and "p_rho_tw0" in site_df.columns:
        d = site_df[["median_Missed_norm", "p_rho_tw0"]].replace([np.inf, -np.inf], np.nan).dropna()
        if len(d) >= 3:
            sp = d["median_Missed_norm"].corr(d["p_rho_tw0"], method="spearman")
            print(f"Spearman(median_Missed_norm, p_rho_tw0)={sp:.4f}")
        else:
            print("Spearman(median_Missed_norm, p_rho_tw0)=NaN (insufficient data)")


def main():
    if not cfg.RUN_INTER_OVERPASS_DIAGNOSTICS:
        print("[volatility] RUN_INTER_OVERPASS_DIAGNOSTICS=False; skipping.")
        return

    out_base = cfg.VOLATILITY_OUTPUT_BASE
    raster_stats = cfg.INTER_OVERPASS_RASTER_STATS

    per_case_site_dfs = []
    per_case_window_dfs = []

    for row_path in cfg.ROW_PATHS:
        for raster_stat in raster_stats:
            window_df, site_df = run_case(
                wit_sms_path=cfg.WIT_SMS_PATH,
                raster_base=cfg.RASTER_BASE,
                row_path=row_path,
                raster_stat=raster_stat,
                min_valid_increments=cfg.MIN_VALID_INCREMENTS,
                epsilon=cfg.EPSILON_MISSED_VARIATION,
                require_complete_path_for_missed=cfg.REQUIRE_COMPLETE_PATH_FOR_MISSED,
                apply_sms_calibration=cfg.VOLATILITY_APPLY_SMS_CALIBRATION,
                apply_rescaling=cfg.VOLATILITY_APPLY_RESCALING,
                temporal_windows_for_corr=cfg.VOLATILITY_TEMPORAL_WINDOWS,
                min_pairs_corr=cfg.VOLATILITY_MIN_PAIRS_CORR,
            )

            out_dir = _case_out_dir(out_base, raster_stat, row_path)
            _save_case_outputs(
                out_dir=out_dir,
                raster_stat=raster_stat,
                row_path_label=row_path,
                window_df=window_df,
                site_df=site_df,
            )
            _print_summary(row_path, raster_stat, window_df, site_df)

            if not site_df.empty:
                per_case_site_dfs.append(site_df)
            if not window_df.empty:
                per_case_window_dfs.append(window_df)

    # Combined case across rows
    if len(per_case_site_dfs) >= 2:
        for raster_stat in raster_stats:
            site_parts = [df for df in per_case_site_dfs if not df.empty and df["raster_stat"].iloc[0] == raster_stat]
            win_parts = [df for df in per_case_window_dfs if not df.empty and df["raster_stat"].iloc[0] == raster_stat]

            if not site_parts:
                continue

            site_df = pd.concat(site_parts, ignore_index=True)
            window_df = pd.concat(win_parts, ignore_index=True) if win_parts else pd.DataFrame()

            combined_label = f"combined_{cfg.ROW_PATHS[0]}_{cfg.ROW_PATHS[1]}"
            out_dir = _case_out_dir(out_base, raster_stat, combined_label)

            _save_case_outputs(
                out_dir=out_dir,
                raster_stat=raster_stat,
                row_path_label=combined_label,
                window_df=window_df,
                site_df=site_df,
            )
            _print_summary(combined_label, raster_stat, window_df, site_df)


if __name__ == "__main__":
    main()