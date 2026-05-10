from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, mannwhitneyu

import config_endp as cfg
from modules.endpoint_module import make_case_diagnostics


def _out_dir(base: Path, raster_stat: str, case_label: str) -> Path:
    d = base / raster_stat / case_label
    d.mkdir(parents=True, exist_ok=True)
    return d


def _save_xlsx(path: Path, sheet_name: str, df: pd.DataFrame):
    path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(path, engine="xlsxwriter") as writer:
        df.to_excel(writer, sheet_name=sheet_name, index=False)


def _scatter_with_fit(df: pd.DataFrame, x: str, y: str, out_png: Path, xlabel: str, ylabel: str, title: str):
    d = df[[x, y]].replace([np.inf, -np.inf], np.nan).dropna()
    if len(d) < 3:
        return

    sp = spearmanr(d[x], d[y], nan_policy="omit")

    fig, ax = plt.subplots(figsize=(6.3, 5.0))
    ax.scatter(d[x], d[y], s=30, alpha=0.85)

    coeffs = np.polyfit(d[x], d[y], 1)
    xs = np.linspace(d[x].min(), d[x].max(), 100)
    ax.plot(xs, coeffs[0] * xs + coeffs[1], linewidth=1.2)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.25)

    txt = f"Spearman = {sp.statistic:.2f}\np = {sp.pvalue:.3g}"
    ax.text(0.03, 0.97, txt, transform=ax.transAxes, va="top", ha="left",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8, edgecolor="0.7"))

    fig.tight_layout()
    fig.savefig(out_png, dpi=350, bbox_inches="tight")
    plt.close(fig)


def _signflip_boxplot(window_df: pd.DataFrame, out_png: Path):
    d = window_df[["endpoint_mismatch_abs", "sign_flip"]].replace([np.inf, -np.inf], np.nan).dropna()
    if d.empty:
        return

    g0 = d[d["sign_flip"] == 0]["endpoint_mismatch_abs"].values
    g1 = d[d["sign_flip"] == 1]["endpoint_mismatch_abs"].values
    if len(g0) < 3 or len(g1) < 3:
        return

    test = mannwhitneyu(g0, g1, alternative="two-sided")

    fig, ax = plt.subplots(figsize=(5.8, 4.8))
    ax.boxplot([g0, g1], tick_labels=["No sign flip", "Sign flip"], showfliers=False)
    ax.set_ylabel(r"Endpoint mismatch $|\Delta y - \Delta x_{end}|$")
    ax.set_title("Endpoint mismatch by change-direction agreement")
    ax.grid(True, axis="y", alpha=0.25)

    txt = (
        f"Median no-flip = {np.nanmedian(g0):.4f}\n"
        f"Median flip = {np.nanmedian(g1):.4f}\n"
        f"Mann-Whitney p = {test.pvalue:.3g}"
    )
    ax.text(0.03, 0.97, txt, transform=ax.transAxes, va="top", ha="left",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8, edgecolor="0.7"))

    fig.tight_layout()
    fig.savefig(out_png, dpi=350, bbox_inches="tight")
    plt.close(fig)


def _tertile_boxplot(site_df: pd.DataFrame, metric: str, out_png: Path, ylabel: str):
    d = site_df[["p90_endpoint_mismatch", metric]].replace([np.inf, -np.inf], np.nan).dropna().copy()
    if len(d) < 9:
        return

    d["group"] = pd.qcut(d["p90_endpoint_mismatch"], 3, duplicates="drop")
    groups = [g[metric].values for _, g in d.groupby("group", observed=False)]
    labels = ["Low", "Mid", "High"]

    fig, ax = plt.subplots(figsize=(6.0, 4.8))
    ax.boxplot(groups, tick_labels=labels, showfliers=False)
    ax.set_xlabel("p90 endpoint mismatch tertile")
    ax.set_ylabel(ylabel)
    ax.set_title(f"{ylabel} across endpoint-mismatch tertiles")
    ax.grid(True, axis="y", alpha=0.25)

    summary = d.groupby("group", observed=False)[metric].mean()
    txt = "\n".join([f"{labels[i]} mean = {summary.iloc[i]:.3f}" for i in range(len(summary))])
    ax.text(0.03, 0.97, txt, transform=ax.transAxes, va="top", ha="left",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8, edgecolor="0.7"))

    fig.tight_layout()
    fig.savefig(out_png, dpi=350, bbox_inches="tight")
    plt.close(fig)


def _summary_tables(window_df: pd.DataFrame, site_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    # Window-level table
    rows_window = []

    for x in ["abs_delta_y", "abs_delta_x_end", "QV_norm", "AV_norm", "window_length_days"]:
        d = window_df[[x, "endpoint_mismatch_abs"]].replace([np.inf, -np.inf], np.nan).dropna()
        if len(d) >= 3:
            sp = spearmanr(d[x], d["endpoint_mismatch_abs"], nan_policy="omit")
            rows_window.append({
                "predictor": x,
                "n": len(d),
                "spearman_r": sp.statistic,
                "spearman_p": sp.pvalue,
            })

    d_sf = window_df[["endpoint_mismatch_abs", "sign_flip"]].replace([np.inf, -np.inf], np.nan).dropna()
    if not d_sf.empty:
        g0 = d_sf[d_sf["sign_flip"] == 0]["endpoint_mismatch_abs"].values
        g1 = d_sf[d_sf["sign_flip"] == 1]["endpoint_mismatch_abs"].values
        if len(g0) >= 3 and len(g1) >= 3:
            mw = mannwhitneyu(g0, g1, alternative="two-sided")
            rows_window.append({
                "predictor": "sign_flip (1 vs 0)",
                "n": len(d_sf),
                "spearman_r": np.nan,
                "spearman_p": np.nan,
                "median_no_flip": float(np.nanmedian(g0)),
                "median_flip": float(np.nanmedian(g1)),
                "mannwhitney_p": float(mw.pvalue),
            })

    window_summary = pd.DataFrame(rows_window)

    # Site-level table
    rows_site = []
    for y in ["p_rho_tw0", "ubrmsd_tw0"]:
        d = site_df[["p90_endpoint_mismatch", y]].replace([np.inf, -np.inf], np.nan).dropna()
        if len(d) >= 3:
            sp = spearmanr(d["p90_endpoint_mismatch"], d[y], nan_policy="omit")
            rows_site.append({
                "response": y,
                "n": len(d),
                "spearman_r": sp.statistic,
                "spearman_p": sp.pvalue,
            })

    d = site_df[["p90_endpoint_mismatch", "p_rho_tw0", "ubrmsd_tw0"]].replace([np.inf, -np.inf], np.nan).dropna().copy()
    if len(d) >= 9:
        d["group"] = pd.qcut(d["p90_endpoint_mismatch"], 3, duplicates="drop")
        tertile = d.groupby("group", observed=False).agg(
            count=("p_rho_tw0", "count"),
            mean_r=("p_rho_tw0", "mean"),
            median_r=("p_rho_tw0", "median"),
            mean_ubrmsd=("ubrmsd_tw0", "mean"),
            median_ubrmsd=("ubrmsd_tw0", "median"),
        ).reset_index()
    else:
        tertile = pd.DataFrame()

    site_summary = pd.DataFrame(rows_site)
    return window_summary, site_summary, tertile


def _save_case(case_label: str, raster_stat: str, window_df: pd.DataFrame, site_df: pd.DataFrame):
    out_dir = _out_dir(cfg.ENDPOINT_OUTPUT_BASE, raster_stat, case_label)

    _save_xlsx(out_dir / f"window_level_endpoint_diagnostics_{raster_stat}_{case_label}.xlsx", "WindowMetrics", window_df)
    _save_xlsx(out_dir / f"site_level_endpoint_diagnostics_{raster_stat}_{case_label}.xlsx", "SiteMetrics", site_df)

    window_summary, site_summary, tertile = _summary_tables(window_df, site_df)
    with pd.ExcelWriter(out_dir / f"endpoint_summary_tables_{raster_stat}_{case_label}.xlsx", engine="xlsxwriter") as writer:
        window_summary.to_excel(writer, sheet_name="WindowLevel", index=False)
        site_summary.to_excel(writer, sheet_name="SiteLevel", index=False)
        tertile.to_excel(writer, sheet_name="Tertiles", index=False)

    if cfg.ENDPOINT_SAVE_PLOTS:
        _scatter_with_fit(
            site_df,
            "p90_endpoint_mismatch",
            "p_rho_tw0",
            out_dir / f"fig_p90_mismatch_vs_prho_{raster_stat}_{case_label}.png",
            xlabel=r"$P90(|\Delta y - \Delta x_{end}|)$",
            ylabel="Pearson r (tw=0)",
            title="Site-level upper-tail endpoint mismatch vs Pearson correlation",
        )

        _scatter_with_fit(
            site_df,
            "p90_endpoint_mismatch",
            "ubrmsd_tw0",
            out_dir / f"fig_p90_mismatch_vs_ubrmsd_{raster_stat}_{case_label}.png",
            xlabel=r"$P90(|\Delta y - \Delta x_{end}|)$",
            ylabel="ubRMSD (tw=0)",
            title="Site-level upper-tail endpoint mismatch vs ubRMSD",
        )

        _signflip_boxplot(
            window_df,
            out_dir / f"fig_signflip_endpoint_mismatch_{raster_stat}_{case_label}.png",
        )

        _tertile_boxplot(
            site_df,
            "p_rho_tw0",
            out_dir / f"fig_tertiles_prho_{raster_stat}_{case_label}.png",
            ylabel="Pearson r (tw=0)",
        )

        _tertile_boxplot(
            site_df,
            "ubrmsd_tw0",
            out_dir / f"fig_tertiles_ubrmsd_{raster_stat}_{case_label}.png",
            ylabel="ubRMSD (tw=0)",
        )


def main():
    if not cfg.RUN_ENDPOINT_DIAGNOSTICS:
        print("[endpoint] RUN_ENDPOINT_DIAGNOSTICS=False; skipping.")
        return

    all_site = []
    all_window = []

    for row_path in cfg.ROW_PATHS:
        for raster_stat in cfg.ENDPOINT_RASTER_STATS:
            window_df, site_df = make_case_diagnostics(
                wit_sms_path=cfg.WIT_SMS_PATH,
                raster_base=cfg.RASTER_BASE,
                row_path=row_path,
                raster_stat=raster_stat,
                min_valid_increments=cfg.ENDPOINT_MIN_VALID_INCREMENTS,
                apply_sms_calibration=cfg.ENDPOINT_APPLY_SMS_CALIBRATION,
                apply_rescaling=cfg.ENDPOINT_APPLY_RESCALING,
                temporal_windows_for_corr=cfg.ENDPOINT_TEMPORAL_WINDOWS,
                min_pairs_corr=cfg.ENDPOINT_MIN_PAIRS_CORR,
            )

            _save_case(row_path, raster_stat, window_df, site_df)

            if not window_df.empty:
                all_window.append(window_df)
            if not site_df.empty:
                all_site.append(site_df)

    if len(all_site) >= 2:
        site_df = pd.concat(all_site, ignore_index=True)
        window_df = pd.concat(all_window, ignore_index=True) if all_window else pd.DataFrame()
        combined_label = f"combined_{cfg.ROW_PATHS[0]}_{cfg.ROW_PATHS[1]}"

        for raster_stat in cfg.ENDPOINT_RASTER_STATS:
            s = site_df[site_df["raster_stat"] == raster_stat].copy()
            w = window_df[window_df["raster_stat"] == raster_stat].copy() if not window_df.empty else pd.DataFrame()
            if not s.empty:
                _save_case(combined_label, raster_stat, w, s)


if __name__ == "__main__":
    main()