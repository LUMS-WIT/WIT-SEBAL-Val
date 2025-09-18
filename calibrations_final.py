"""
4.
Takes in the merged calibration dataset and applies energy balance corrections. based on Bowen ratio, residual-LE
and non-closure

Goal:  Regression on EF and Correction on LE

Returns
-------
pd.DataFrame
    A dataframe containing the corrected energy flux estimates. and alpha and beta
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# -----------------------------
# Inputs & Outputs
# -----------------------------
merged_csv = r"merged_calibration_multisite.csv"   # merged SEBAL + Tower dataset
out_dir    = r"./ef_calibration"
os.makedirs(out_dir, exist_ok=True)

params_json = os.path.join(out_dir, "global_EF_params.json")
report_csv  = os.path.join(out_dir, "EF_calibration_report.csv")
out_csv     = os.path.join(out_dir, "merged_calibration_EF_corrected.csv")

# -----------------------------
# Calibration options
# -----------------------------
TOWER_EF_CLOSURE_MODE = "bowen"   # ### CHANGED/NEW: choose "residual" | "bowen" | "le_over_a"
A_MIN = 1e-6                          # ### CHANGED/NEW: avoid divide-by-zero in At
EF_CLIP = (0.0, 1.8)                  # ### CHANGED/NEW: optional physical bounds for EF at L8 overpass (~10:30)
# EF_CLIP = None                  # ### CHANGED/NEW: optional physical bounds for EF at L8 overpass (~10:30)
# -----------------------------
# 1. Load data
# -----------------------------
df = pd.read_csv(merged_csv)

# Ensure numeric (include EF_inst_avg if present)
for col in ["Rn_inst_avg","G_inst_avg","LE_inst_avg",
            "NETRAD","G_F_MDS","LE_F_MDS","H_F_MDS","EF_inst_avg"]:  # ### CHANGED/NEW: add H_F_MDS
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

# -----------------------------
# 1b. Filtering rules (row-wise remove)
# -----------------------------
mask_valid = df[["NETRAD","G_F_MDS","LE_F_MDS"]].notna().all(axis=1)   # tower must exist
mask_valid &= df["Rn_inst_avg"] >= 0                                   # no negative Rn (model)
mask_valid &= (df["LE_inst_avg"] <= 800) & (df["LE_inst_avg"] >= -400) # practical LE bounds

before = len(df)
df = df[mask_valid].copy()
after = len(df)
print(f"✅ Filtering complete: removed {before - after} rows, kept {after}")

# -----------------------------
# 2. Available energy (Tower only; Model A_m optional/diagnostic)
# -----------------------------
A_t = df["NETRAD"] - df["G_F_MDS"]              # tower available energy (closure on tower)
A_m = df["Rn_inst_avg"] - df["G_inst_avg"]      # optional/diagnostic only (NOT used to correct LE)

# -----------------------------
# 3. TOWER EF with explicit closure   ### CHANGED/NEW (core change)
# -----------------------------
# Residual-LE closure:   LE*_t = A_t - H_t,     EF*_t = LE*_t / A_t
# Bowen-ratio closure:   beta_t = H_t/LE_t,     EF*_t = 1/(1+beta_t)
# Simple normalized (fallback):           EF_t = LE_t / A_t

def compute_tower_EF_closed(df, mode):
    At = A_t.copy()
    At = At.where(np.isfinite(At) & (At > A_MIN), np.nan)

    if mode == "residual":
        if "H_F_MDS" in df.columns and df["H_F_MDS"].notna().any():
            LE_res = At - df["H_F_MDS"]
            EF_star = LE_res / At
            used = "residual-LE"
        else:
            # fallback to LE/A
            EF_star = df["LE_F_MDS"] / At
            used = "LE/A (fallback; H_F_MDS missing)"
    elif mode == "bowen":
        if "H_F_MDS" in df.columns and df["H_F_MDS"].notna().any() and df["LE_F_MDS"].notna().any():
            beta_t = df["H_F_MDS"] / df["LE_F_MDS"]
            EF_star = 1.0 / (1.0 + beta_t)
            used = "Bowen-ratio"
        else:
            # fallback to LE/A
            EF_star = df["LE_F_MDS"] / At
            used = "LE/A (fallback; need H and LE)"
    else:
        EF_star = df["LE_F_MDS"] / At
        used = "LE/A (no explicit closure)"

    return EF_star, used

EF_t, tower_mode_used = compute_tower_EF_closed(df, TOWER_EF_CLOSURE_MODE)
print(f"Tower EF closure mode used: {tower_mode_used}")

# -----------------------------
# 4. MODEL EF (no model closure)
# -----------------------------
# Prefer EF_inst_avg if present; otherwise (as last resort) LE/A_m
if "EF_inst_avg" in df.columns and df["EF_inst_avg"].notna().any():
    EF_m = df["EF_inst_avg"].copy()
else:
    EF_m = df["LE_inst_avg"] / A_m
    print("⚠️ EF_inst_avg not found/empty; falling back to EF_m = LE_m / (Rn_m - G_m).")

# Optional: clip EF to a physical range to stabilize fit
if EF_CLIP is not None:
    lo, hi = EF_CLIP
    EF_t = EF_t.clip(lo, hi)
    EF_m = EF_m.clip(lo, hi)

# -----------------------------
# 5. Fit EF calibration (tower ~ model)
# -----------------------------
mask = np.isfinite(EF_m) & np.isfinite(EF_t)
print("Valid calibration pairs (EF):", int(mask.sum()))

if mask.sum() >= 3:
    reg = LinearRegression()
    reg.fit(EF_m[mask].to_numpy().reshape(-1,1), EF_t[mask].to_numpy())
    alpha, beta = float(reg.coef_[0]), float(reg.intercept_)
else:
    alpha, beta = 1.0, 0.0  # fallback (no calibration)

print(f" EF_tower (closed) ≈ {alpha:.3f} * EF_model + {beta:.3f}")

# -----------------------------
# 6. Apply EF calibration to model LE (ratio-based; NO model closure)  ### CHANGED/NEW
# -----------------------------
EF_corr = alpha * EF_m + beta
if EF_CLIP is not None:
    lo, hi = EF_CLIP
    EF_corr = EF_corr.clip(lo, hi)

# EF-ratio scaling (no use of A_m):
# LE_corr = LE_model * (EF_corr / EF_model)
ratio = np.where(np.isfinite(EF_m) & (EF_m != 0), EF_corr / EF_m, np.nan)
LE_corr_EF = df["LE_inst_avg"] * ratio

# Optional diagnostic H using model Rn and G (diagnostic only)
H_corr_EF  = (df["Rn_inst_avg"] - df["G_inst_avg"]) - LE_corr_EF

# -----------------------------
# 7. Save calibration params
# -----------------------------
params = {
    "tower_EF_closure_mode_requested": TOWER_EF_CLOSURE_MODE,
    "tower_EF_closure_mode_used": tower_mode_used,
    "EF_map": {"alpha": alpha, "beta": beta, "EF_clip": EF_CLIP},
    "notes": "Tower EF closed; model EF from EF_inst_avg (preferred). LE corrected via EF-ratio (no model closure)."
}
with open(params_json, "w") as f:
    json.dump(params, f, indent=2)

# -----------------------------
# 8. Evaluation on LE (before vs after)
# -----------------------------
def stats(y_true, y_pred):
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if mask.sum() < 3:
        return np.nan, np.nan, np.nan
    err = y_pred[mask] - y_true[mask]
    rmse = float(np.sqrt(np.mean(err**2)))
    bias = float(np.mean(err))
    r2 = float(np.corrcoef(y_true[mask], y_pred[mask])[0,1]**2)
    return rmse, r2, bias

LE_tower = df["LE_F_MDS"]
rmse_b, r2_b, bias_b = stats(LE_tower, df["LE_inst_avg"])  # before
rmse_a, r2_a, bias_a = stats(LE_tower, LE_corr_EF)         # after EF

report = pd.DataFrame({
    "metric": ["RMSE", "R2", "Bias"],
    "EF_before": [rmse_b, r2_b, bias_b],
    "EF_after":  [rmse_a, r2_a, bias_a],
})
report.to_csv(report_csv, index=False)
print(report)

# -------------------------------------------------------
# Detailed report with per-site counts and dates used
# -------------------------------------------------------
mask_used = np.isfinite(EF_m) & np.isfinite(EF_t)
cols_keep = []
for c in ["site", "scene", "date", "LE_inst_avg", "LE_F_MDS",
          "Rn_inst_avg", "G_inst_avg", "NETRAD", "G_F_MDS", "H_F_MDS"]:
    if c in df.columns:
        cols_keep.append(c)

df_pairs = df.loc[mask_used, cols_keep].copy()
df_pairs["EF_model"] = EF_m[mask_used].values
df_pairs["EF_tower_closed"] = EF_t[mask_used].values

if "date" in df_pairs.columns:
    df_pairs["date"] = pd.to_datetime(df_pairs["date"], errors="coerce").dt.date
if "site" not in df_pairs.columns:
    df_pairs["site"] = "ALL"

rows = []
for site, g in df_pairs.groupby("site"):
    if "date" in g.columns:
        dates_list = sorted([str(d) for d in pd.Series(g["date"]).dropna().unique()])
        dates_str = ", ".join(dates_list)
    else:
        dates_str = ""
    rows.append({"site": site, "n_points": int(len(g)), "dates_used": dates_str})
by_site = pd.DataFrame(rows).sort_values("site")

total_points = int(df_pairs.shape[0])
summary = pd.DataFrame([{"total_points_used": total_points,
                         "tower_closure_mode_used": tower_mode_used}])

detailed_xlsx = os.path.join(out_dir, "EF_calibration_report_detailed.xlsx")
with pd.ExcelWriter(detailed_xlsx, engine="xlsxwriter") as writer:
    report.to_excel(writer, sheet_name="metrics", index=False)
    by_site.to_excel(writer, sheet_name="by_site", index=False)
    df_pairs.to_excel(writer, sheet_name="pairs", index=False)
    summary.to_excel(writer, sheet_name="summary", index=False)

print("✅ Detailed report written:")
print(f"  - Metrics CSV: {report_csv}")
print(f"  - Detailed Excel: {detailed_xlsx}")
print(f"  - Total points used: {total_points}")
print("  - Per-site counts:")
print(by_site[["site", "n_points"]])

# -----------------------------
# Save corrected dataset
# -----------------------------
df_out = df.copy()
df_out["EF_model"]   = EF_m
df_out["EF_tower_closed"] = EF_t
df_out["EF_corr"]    = EF_corr
df_out["LE_corr_EF"] = LE_corr_EF
df_out["H_corr_EF"]  = H_corr_EF
df_out.to_csv(out_csv, index=False, float_format="%.3f")
print(f"✅ Corrected dataset saved: {out_csv}")

# -----------------------------
# Plots (LE before/after)
# -----------------------------
def scatter_plot(x, y, title, fname, rmse, r2, bias):
    plt.figure(figsize=(6,6))
    plt.scatter(x, y, c="blue", alpha=0.6, edgecolor="k")
    min_val = np.nanmin([np.nanmin(x), np.nanmin(y)])
    max_val = np.nanmax([np.nanmax(x), np.nanmax(y)])
    plt.plot([min_val, max_val], [min_val, max_val], "r--", label="1:1")
    plt.xlabel("LE (Model) [W/m²]")
    plt.ylabel("LE (Tower) [W/m²]")
    plt.title(f"{title}  [{tower_mode_used}]")
    plt.text(0.05, 0.95, f"RMSE={rmse:.2f} W/m²\nR²={r2:.2f}\nBias={bias:.2f} W/m²",
             transform=plt.gca().transAxes, va="top", ha="left",
             bbox=dict(facecolor="white", alpha=0.7))
    plt.legend(); plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout(); plt.savefig(os.path.join(out_dir, fname), dpi=300); plt.close()

scatter_plot(df["LE_inst_avg"], LE_tower, "Latent Heat (LE) Before Calibration", "LE_before.png", rmse_b, r2_b, bias_b)
scatter_plot(LE_corr_EF,        LE_tower, "Latent Heat (LE) After Calibration", "LE_after_EF.png", rmse_a, r2_a, bias_a)

# -----------------------------
# Calibration plot (EF_t_closed vs EF_m)
# -----------------------------
def calib_scatter(x, y, slope, intercept, xlabel, ylabel, title, fname):
    plt.figure(figsize=(6,6))
    plt.scatter(x, y, c="blue", alpha=0.6, edgecolor="k")
    min_val = np.nanmin([np.nanmin(x), np.nanmin(y)])
    max_val = np.nanmax([np.nanmax(x), np.nanmax(y)])
    plt.plot([min_val, max_val], [min_val, max_val], "r--", label="1:1")
    xx = np.linspace(min_val, max_val, 100)
    yy = slope * xx + intercept
    plt.plot(xx, yy, "g-", label=f"Fit: y = {slope:.3f}x + {intercept:.3f}")
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() > 2:
        y_hat = slope * x[mask] + intercept
        rmse = np.sqrt(np.mean((y_hat - y[mask])**2))
        bias = np.mean(y_hat - y[mask])
        r2 = np.corrcoef(y[mask], y_hat)[0,1]**2
        txt = f"RMSE={rmse:.2f}\nR²={r2:.2f}\nBias={bias:.2f}"
    else:
        txt = "insufficient pairs"
    plt.xlabel(xlabel); plt.ylabel(ylabel); plt.title(title + f" [{tower_mode_used}]")
    plt.text(0.05, 0.95, txt, transform=plt.gca().transAxes,
             va="top", ha="left", bbox=dict(facecolor="white", alpha=0.7))
    plt.legend(); plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout(); plt.savefig(os.path.join(out_dir, fname), dpi=300); plt.close()

calib_scatter(
    EF_m, EF_t,
    slope=alpha, intercept=beta,
    xlabel="EF (Model)", ylabel="EF (Tower, closed)",
    title="EF Calibration",
    fname="EF_calibration.png"
)



print(f"✅ Outputs:\n- Params: {params_json}\n- Report: {report_csv}\n- CSV: {out_csv}\n- Plots in: {out_dir}")



