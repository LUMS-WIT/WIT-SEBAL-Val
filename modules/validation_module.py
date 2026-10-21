"""Shared matching-window and plotting helpers for held-out validation.

The retired same-data rescaling driver is archived locally. These helpers retain
its plotting conventions and support the finalized workflow.
"""
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import metrics


def tw_to_window_params(tw: int):
    """
    tw is in days (0, 3, 5, 7).
    Returns (half_window, min_valid).
    """
    if tw == 0:
        return 0, 1
    if tw == 3:
        return 1, 2
    if tw == 5:
        return 2, 3
    if tw == 7:
        return 3, 4
    raise ValueError(f"Unsupported TEMPORAL_WIN={tw}. Use one of: 0, 3, 5, 7.")


def make_paired_scatter_figure(paired_values: pd.DataFrame, stats_results: dict, title: str,
                             metrics_override=None):
    x = paired_values["sebal_sm"].to_numpy(dtype=float)
    y = paired_values["wit_sm"].to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    ax.scatter(x, y, s=12, alpha=0.5, edgecolors="none")

    if x.size and y.size:
        min_v = float(min(x.min(), y.min()))
        max_v = float(max(x.max(), y.max()))
        ax.plot([min_v, max_v], [min_v, max_v], "--", color="gray", linewidth=1)

    ax.set_xlabel("SEBAL-derived soil moisture (m³/m³)")
    ax.set_ylabel("WITSMS soil moisture (m³/m³)")
    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal", adjustable="box")


    if metrics_override is None:
        # Keep the legacy callers' calculations unchanged.
        bias = metrics.bias(x, y)
        mse, mse_corr, mse_bias, mse_var = metrics.mse(x, y)
        ubrmsd = metrics.ubrmsd(x, y)
        p_rho = metrics.pearson_r(x, y)
        s_rho = metrics.spearman_r(x, y)
    else:
        # The scaling workflow retains short/constant series: errors remain
        # defined even when a correlation cannot be estimated.
        bias, mse, ubrmsd, p_rho, s_rho = (
            metrics_override[key] for key in ["bias", "MSE", "ubRMSD", "Pearson", "Spearman"]
        )



    rmse = (mse ** 0.5) if mse is not None else float("nan")
    textstr = (
        f"Bias = {bias:.3f} m³/m³\n"
        f"MSE = {mse:.3f} (m³/m³)²\n"
        f"ubRMSD = {ubrmsd:.3f} m³/m³\n"
        f"R = {p_rho:.3f}\n"
        fr"$\rho$ = {s_rho:.3f}"
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
