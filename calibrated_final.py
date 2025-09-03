"""
4b
Once calibrated, 
This script generates scatter plots and site-level statistics for energy flux (EF) estimates.

Returns
-------
None
    This script saves figures and CSV files to the specified output paths.  

"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
from pathlib import Path

# -----------------------------
# Inputs/outputs
# -----------------------------
csv_in         = r"merged_calibration_multisite.csv"  # <- change if needed
pairs_out      = "calibrated/ef_tower_model_pairs.csv"
fig_overall    = "calibrated/ef_scatter.png"
fig_site       = "calibrated/ef_site_means.png"
site_stats_out = "calibrated/ef_site_stats.csv"

# === NEW (LE) ===
fig_overall_LE = "calibrated/le_scatter.png"
fig_site_LE    = "calibrated/le_site_means.png"

EF_MAX = 1.2

# -----------------------------
# Load
# -----------------------------
df = pd.read_csv(csv_in)

def pick(df, *cands):
    cols = {c.lower(): c for c in df.columns}
    for cand in cands:
        c = cand.lower()
        if c in cols:
            return cols[c]
    return None

# Tower (from your file)
col_Rn_t = pick(df, "NETRAD")
col_G_t  = pick(df, "G_F_MDS")
col_LE_t = pick(df, "LE_F_MDS")
col_H_t  = pick(df, "H_F_MDS")  # not used here

# Model (use *_inst_avg in your file)
col_Rn_m = pick(df, "Rn_inst_avg")
col_G_m  = pick(df, "G_inst_avg")
col_LE_m = pick(df, "LE_inst_avg")
col_H_m  = pick(df, "H_inst_avg")  # not used here

required = [col_Rn_t, col_G_t, col_LE_t, col_Rn_m, col_G_m, col_LE_m]
if any(c is None for c in required):
    missing = [n for n, c in zip(
        ["NETRAD","G_F_MDS","LE_F_MDS","Rn_inst_avg","G_inst_avg","LE_inst_avg"],
        required
    ) if c is None]
    raise ValueError(f"Missing required columns: {missing}. Found: {list(df.columns)}")

def safe_div(numer, denom):
    numer = np.asarray(numer, dtype=float)
    denom = np.asarray(denom, dtype=float)
    out = np.full_like(numer, np.nan, dtype=float)
    mask = np.isfinite(numer) & np.isfinite(denom) & (denom > 0.0)
    out[mask] = numer[mask] / denom[mask]
    return out

# -----------------------------
# Compute EF = LE / (Rn - G)
# -----------------------------
rn_t = df[col_Rn_t].astype(float).values
g_t  = df[col_G_t].astype(float).values
le_t = df[col_LE_t].astype(float).values
rn_t[rn_t <= 0] = np.nan  # net radiation cannot be <= 0
EF_tower = safe_div(le_t, rn_t - g_t)

rn_m = df[col_Rn_m].astype(float).values
g_m  = df[col_G_m].astype(float).values
le_m = df[col_LE_m].astype(float).values
rn_m[rn_m <= 0] = np.nan
EF_model = safe_div(le_m, rn_m - g_m)

# Pair, clean
paired = pd.DataFrame({"EF_tower": EF_tower, "EF_model": EF_model, "site": df.get("site")})
paired = paired.replace([np.inf, -np.inf], np.nan).dropna()

# Optional EF range filter (applied to keep EF-based comparability)
paired = paired[(paired["EF_tower"].between(0, EF_MAX)) &
                (paired["EF_model"].between(0, EF_MAX))]

if paired.empty:
    raise ValueError("No valid EF pairs after filtering. Check inputs.")

# -----------------------------
# Overall metrics & plot (EF)
# -----------------------------
overall_rmse = float(np.sqrt(mean_squared_error(paired["EF_tower"], paired["EF_model"])))
overall_r2   = float(r2_score(paired["EF_tower"], paired["EF_model"]))
paired[["EF_tower","EF_model","site"]].to_csv(pairs_out, index=False)

plt.figure(figsize=(6,6))
x = paired["EF_tower"].values
y = paired["EF_model"].values
plt.scatter(x, y, s=16, alpha=0.8)

mn = float(np.nanmin([x.min(), y.min(), 0.0]))
mx = float(np.nanmax([x.max(), y.max(), 1.0]))
plt.plot([mn, mx], [mn, mx], "--", linewidth=1)

plt.xlabel("EF (Tower)")
plt.ylabel("EF (Model)")
plt.title("Evaporative Fraction (Model vs Tower)")

txt = f"RMSE = {overall_rmse:.3f}\nR² = {overall_r2:.3f}\nN = {len(paired)}"
plt.text(mn + 0.05*(mx-mn), mn + 0.90*(mx-mn), txt,
         bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

plt.xlim(mn, mx); plt.ylim(mn, mx)
plt.gca().set_aspect("equal", adjustable="box")
plt.grid(True, linestyle=":", linewidth=0.6, alpha=0.7)
plt.tight_layout()
plt.savefig(fig_overall, dpi=200)
plt.close()

# -----------------------------
# Site-wise stats (EF means, std, N)
# -----------------------------
site_stats = (
    paired.groupby("site", dropna=True)
          .agg(
              EF_tower_mean=("EF_tower","mean"),
              EF_tower_std =("EF_tower","std"),
              EF_model_mean=("EF_model","mean"),
              EF_model_std =("EF_model","std"),
              N=("EF_tower","size")
          ).reset_index()
)

# Metrics on site means (EF) for plot annotation
x_means = site_stats["EF_tower_mean"].values
y_means = site_stats["EF_model_mean"].values
rmse_sites = float(np.sqrt(mean_squared_error(x_means, y_means)))
try:
    r2_sites = float(r2_score(x_means, y_means))
except Exception:
    r2_sites = np.nan

# -----------------------------
# Site-wise plot (EF) with whiskers, counts, metrics
# -----------------------------
plt.figure(figsize=(6,6))
xmin = float(np.nanmin([0.0, (site_stats["EF_tower_mean"] - site_stats["EF_tower_std"]).min()]))
xmax = float(np.nanmax([
    1.0,
    (site_stats["EF_tower_mean"] + site_stats["EF_tower_std"]).max(),
    (site_stats["EF_model_mean"] + site_stats["EF_model_std"]).max()
]))
mn, mx = xmin, xmax
plt.plot([mn, mx], [mn, mx], "--", linewidth=1)

for _, r in site_stats.iterrows():
    label_ = f"({int(r['N'])})"
    label = f"{r['site']}"
    plt.errorbar(
        r["EF_tower_mean"], r["EF_model_mean"],
        xerr=r["EF_tower_std"], yerr=r["EF_model_std"],
        fmt="o", capsize=4, elinewidth=1, alpha=0.95, label=label
    )
    plt.annotate(label_, (r["EF_tower_mean"], r["EF_model_mean"]),
                 xytext=(6, 6), textcoords="offset points")

txt_site = (f"RMSE = {rmse_sites:.3f}\n"
            f"R² = {r2_sites:.3f}")
plt.text(mn + 0.05*(mx-mn), mn + 0.90*(mx-mn), txt_site,
         bbox=dict(boxstyle="round", facecolor="white", alpha=0.85))

plt.xlabel("EF mean (Tower)")
plt.ylabel("EF mean (Model)")
plt.title("Site-wise Mean EF ± 1σ")
plt.xlim(mn, mx); plt.ylim(mn, mx)
plt.gca().set_aspect("equal", adjustable="box")
plt.grid(True, linestyle=":", linewidth=0.6, alpha=0.7)
plt.legend(loc="best", frameon=True)
plt.tight_layout()
plt.savefig(fig_site, dpi=200)
plt.close()

# ======================================================================
# === NEW (LE): site-specific RMSE/R² and plots, plus CSV augmentation ==
# ======================================================================

# Use the SAME rows as EF analysis for comparability
paired["LE_tower"] = le_t[paired.index]
paired["LE_model"] = le_m[paired.index]
paired = paired.replace([np.inf, -np.inf], np.nan).dropna(subset=["LE_tower","LE_model","site"])

# --- Overall LE scatter (kept) ---
plt.figure(figsize=(6,6))
xL = paired["LE_tower"].values
yL = paired["LE_model"].values
plt.scatter(xL, yL, s=16, alpha=0.8)
mnL = float(np.nanmin([xL.min(), yL.min(), 0.0]))
mxL = float(np.nanmax([xL.max(), yL.max()]))
plt.plot([mnL, mxL], [mnL, mxL], "--", linewidth=1)
plt.xlabel("LE (Tower) [W m⁻²]")
plt.ylabel("LE (Model) [W m⁻²]")
plt.title("Latent Heat (Model vs Tower)")

overall_rmse_LE = float(np.sqrt(mean_squared_error(xL, yL)))
try:
    overall_r2_LE = float(r2_score(xL, yL))
except Exception:
    overall_r2_LE = np.nan

txtL = f"RMSE = {overall_rmse_LE:.1f}\nR² = {overall_r2_LE:.3f}\nN = {len(paired)}"
plt.text(mnL + 0.05*(mxL-mnL), mnL + 0.90*(mxL-mnL), txtL,
         bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
plt.xlim(mnL, mxL); plt.ylim(mnL, mxL)
plt.gca().set_aspect("equal", adjustable="box")
plt.grid(True, linestyle=":", linewidth=0.6, alpha=0.7)
plt.tight_layout()
plt.savefig(fig_overall_LE, dpi=200)
plt.close()

# --- Site-wise LE stats from the same paired rows ---
site_stats_LE = (
    paired.groupby("site", dropna=True)
          .agg(
              LE_tower_mean=("LE_tower","mean"),
              LE_tower_std =("LE_tower","std"),
              LE_model_mean=("LE_model","mean"),
              LE_model_std =("LE_model","std")
          ).reset_index()
)

# Merge LE stats into EF table
site_stats = pd.merge(site_stats, site_stats_LE, on="site", how="left")

# === NEW (CSV): per-site RMSE via site means (EF & LE)
site_stats["EF_rmse_mean"] = np.sqrt((site_stats["EF_model_mean"] - site_stats["EF_tower_mean"])**2)
site_stats["LE_rmse_mean"] = np.sqrt((site_stats["LE_model_mean"] - site_stats["LE_tower_mean"])**2)

# === NEW (CSV): LE relative error (%) = RMSE(LE_mean) / LE_tower_mean * 100
with np.errstate(divide="ignore", invalid="ignore"):
    site_stats["LE_rel_error_pct"] = np.where(
        site_stats["LE_tower_mean"] != 0,
        (site_stats["LE_rmse_mean"] / site_stats["LE_tower_mean"]) * 100.0,
        np.nan
    )

# === NEW (CSV): reorder so N is 2nd column, then save with 3-dp precision
ordered_cols = [
    "site", "N",
    "EF_tower_mean", "EF_model_mean", "EF_rmse_mean", "EF_tower_std", "EF_model_std",
    "LE_tower_mean", "LE_model_mean", "LE_rmse_mean", "LE_rel_error_pct", "LE_tower_std", "LE_model_std"
]
ordered_cols += [c for c in site_stats.columns if c not in ordered_cols]
site_stats = site_stats[ordered_cols]
site_stats.to_csv(site_stats_out, index=False, float_format="%.3f")

# --- Site-wise LE plot (kept) ---
plt.figure(figsize=(6,6))
xminL = float(np.nanmin([0.0, (site_stats["LE_tower_mean"] - site_stats["LE_tower_std"]).min()]))
xmaxL = float(np.nanmax([
    (site_stats["LE_tower_mean"] + site_stats["LE_tower_std"]).max(),
    (site_stats["LE_model_mean"] + site_stats["LE_model_std"]).max()
]))
mnL, mxL = xminL, xmaxL
plt.plot([mnL, mxL], [mnL, mxL], "--", linewidth=1)

for _, r in site_stats.iterrows():
    label_ = f"({int(r['N'])})"
    label = f"{r['site']}"
    plt.errorbar(
        r["LE_tower_mean"], r["LE_model_mean"],
        xerr=r["LE_tower_std"], yerr=r["LE_model_std"],
        fmt="o", capsize=4, elinewidth=1, alpha=0.95, label=label
    )
    plt.annotate(label_, (r["LE_tower_mean"], r["LE_model_mean"]),
                 xytext=(6, 6), textcoords="offset points")

# Metrics on site means (LE) for plot annotation
x_means_LE = site_stats["LE_tower_mean"].values
y_means_LE = site_stats["LE_model_mean"].values
rmse_sites_LE = float(np.sqrt(mean_squared_error(x_means_LE, y_means_LE)))
try:
    r2_sites_LE = float(r2_score(x_means_LE, y_means_LE))
except Exception:
    r2_sites_LE = np.nan

txt_site_LE = (f"RMSE = {rmse_sites_LE:.1f}\n"
               f"R² = {r2_sites_LE:.3f}")
plt.text(mnL + 0.05*(mxL-mnL), mnL + 0.90*(mxL-mnL), txt_site_LE,
         bbox=dict(boxstyle="round", facecolor="white", alpha=0.85))

plt.xlabel("LE mean (Tower) [W m⁻²]")
plt.ylabel("LE mean (Model) [W m⁻²]")
plt.title("Site-wise Mean LE ± 1σ")
plt.xlim(mnL, mxL); plt.ylim(mnL, mxL)
plt.gca().set_aspect("equal", adjustable="box")
plt.grid(True, linestyle=":", linewidth=0.6, alpha=0.7)
plt.legend(loc="best", frameon=True)
plt.tight_layout()
plt.savefig(fig_site_LE, dpi=200)
plt.close()

print(f"Saved: {Path(fig_overall).resolve()}")
print(f"Saved: {Path(fig_site).resolve()}")
print(f"Saved: {Path(fig_overall_LE).resolve()}")
print(f"Saved: {Path(fig_site_LE).resolve()}")
print(f"Saved: {Path(pairs_out).resolve()}")
print(f"Saved: {Path(site_stats_out).resolve()}")
