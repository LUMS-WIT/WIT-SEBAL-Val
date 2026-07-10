import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

FOLDER_PATH = Path(r"validations_Output\results")

# -----------------------------
# 1. File paths
# -----------------------------
files = {
    0: "validations_tw_0.xlsx",
    3: "validations_tw_3.xlsx",
    5: "validations_tw_5.xlsx",
    7: "validations_tw_7.xlsx",
}

# -----------------------------
# 2. Read summary files
# -----------------------------
records = []

for window, fp in files.items():
    fp = FOLDER_PATH / fp
    df = pd.read_excel(fp, sheet_name="Summary")
    df = df.set_index("Metric")

    rec = {
        "window": window,
        "N": int(df.loc["Observations", "mean"]),
        "bias_mean": float(df.loc["bias", "mean"]),
        "mse_mean": float(df.loc["mse", "mean"]),
        "ubrmsd_mean": float(df.loc["ubrmsd", "mean"]),
        "pearson_mean": float(df.loc["p_rho", "mean"]),
        "spearman_mean": float(df.loc["s_rho", "mean"]),
        "pearson_median": float(df.loc["p_rho", "median"]),
        "spearman_median": float(df.loc["s_rho", "median"]),
        "pearson_iqr": float(df.loc["p_rho", "IQR"]),
        "spearman_iqr": float(df.loc["s_rho", "IQR"]),
    }

    rec["abs_bias_mean"] = abs(rec["bias_mean"])
    rec["rmse_mean"] = np.sqrt(rec["mse_mean"])

    records.append(rec)

summary = pd.DataFrame(records).sort_values("window").reset_index(drop=True)

# Optional: print the table used for plotting
print("\nSummary used for plotting:\n")
print(
    summary[
        [
            "window", "N",
            "pearson_mean", "spearman_mean",
            "abs_bias_mean", "ubrmsd_mean", "rmse_mean",
            "pearson_median", "spearman_median"
        ]
    ].round(4)
)

# -----------------------------
# 3. Publication-style plotting setup
# -----------------------------
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.labelsize": 10,
    "axes.titlesize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.dpi": 150,
    "savefig.dpi": 600,
    "axes.linewidth": 0.8,
})

x = summary["window"].values

fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.8), constrained_layout=True)

# -----------------------------
# 4. Panel A: Correlations
# -----------------------------
ax = axes[0]

ax.plot(
    x, summary["pearson_mean"],
    marker="o", markersize=5, linewidth=1.6,
    color="black", label="Pearson r"
)
ax.plot(
    x, summary["spearman_mean"],
    marker="s", markersize=5, linewidth=1.6,
    linestyle="--", color="0.35", label="Spearman ρ"
)

# Annotate N above the higher of the two curves
for xi, yi1, yi2, n in zip(x, summary["pearson_mean"], summary["spearman_mean"], summary["N"]):
    y_text = max(yi1, yi2) + 0.018
    ax.annotate(
        f"N={n}",
        xy=(xi, y_text),
        ha="center", va="bottom",
        fontsize=8
    )

ax.set_xlabel("Temporal window (days)")
ax.set_ylabel("Correlation")
ax.set_xticks(x)
ax.set_ylim(0.45, 1.02)
ax.legend(frameon=False, loc="lower right")
ax.set_title("(a) Correlation sensitivity to temporal aggregation window")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# -----------------------------
# 5. Panel B: Error metrics
#    MSE is converted to RMSE so all metrics are in same units
# -----------------------------
ax = axes[1]

# ax.plot(
#     x, summary["abs_bias_mean"],
#     marker="^", markersize=5, linewidth=1.4,
#     color="black", label="|Bias|"
# )
ax.plot(
    x, summary["ubrmsd_mean"],
    marker="o", markersize=5, linewidth=1.4,
    color="0.35", label="ubRMSD"
)
ax.plot(
    x, summary["rmse_mean"],
    marker="s", markersize=5, linewidth=1.4,
    linestyle="--", color="0.60", label="RMSE"
)

# Annotate N near RMSE curve
for xi, yi, n in zip(x, summary["rmse_mean"], summary["N"]):
    ax.annotate(
        f"N={n}",
        xy=(xi, yi + 0.0010),
        ha="center", va="bottom",
        fontsize=8
    )

ax.set_xlabel("Temporal aggregation window (days)")
ax.set_ylabel(r"Error ($m^3\,m^{-3}$)")
ax.set_xticks(x)
ax.legend(frameon=False, loc="upper right")
ax.set_title("(b) Error sensitivity to temporal aggregation window")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# -----------------------------
# 6. Save figure
# -----------------------------
# out_png = "temporal_window_sensitivity.png"
# out_pdf = "temporal_window_sensitivity.pdf"

# fig.savefig(out_png, bbox_inches="tight")
# fig.savefig(out_pdf, bbox_inches="tight")

# plt.show()


##################################################################
# Code for fractional windows sign flip analysis
##################################################################


path = "./endpoint_diagnostics_Output/mean/combined_149039_150039/site_level_endpoint_diagnostics_mean_combined_149039_150039.xlsx"
df = pd.read_excel(path)
df = df[["signflip_fraction", "p_rho_tw0"]].dropna().copy()
# Custom bins requested
bins = [0.25, 0.50, 0.75, 1.00]
labels = ["[0.25, 0.50]", "(0.50, 0.75]", "(0.75, 1.00]"]
df["signflip_bin"] = pd.cut(df["signflip_fraction"], bins=bins, labels=labels, include_lowest=True)
plot_df = df.dropna(subset=["signflip_bin"]).copy()
fig, ax = plt.subplots(figsize=(7,5))
groups = [plot_df.loc[plot_df["signflip_bin"] == lab, "p_rho_tw0"].values for lab in labels]
ax.boxplot(groups, tick_labels=labels)
ax.set_xlabel("Fraction of satellite-overpass windows with polarity disagreement")
ax.set_ylabel("Pearson r")
ax.grid(True, axis="y", alpha=0.3)
plt.xticks(rotation=15, ha="right")
plt.tight_layout()
# out = "/mnt/data/boxplot_r_by_signflip_requestedbins.png"
# plt.savefig(out, dpi=200, bbox_inches="tight")
plt.show()
print(plot_df["signflip_bin"].value_counts().reindex(labels))


# Counts per bin:
# [0.25, 0.50]: 48 sites
# (0.50, 0.75]: 11 sites
# (0.75, 1.00]: 22 sites